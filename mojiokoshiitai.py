import threading
from queue import Queue
import struct
import time

import numpy as np
import pyaudio
import speech_recognition as sr


SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP_THRESHOLD = 2000  # 音声の有無の閾値
LOW_COUNT_MAX = 3  # 連続して閾値を下回った回数の上限
DEVICE_INDEX = 1  # マイクのデバイス番号
PRINT_DURATION = 0.05  # 1文字表示する間隔（秒）
FINISH_TEXT = "おしまい"  # 録音終了のキーワード


def list_audio_devices() -> None:
    """ マイクのデバイス番号を表示 """
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"Device ID {i}: {info['name']}")
    audio.terminate()


def print_gradually(texts_q: Queue) -> None:
    """ 1文字ずつ間をあけて表示

    Args:
        texts_q (Queue): 表示する文字列を格納したキュー
    """
    while True:
        text = texts_q.get()
        if text is None:
            texts_q.task_done()
            break

        print(text, end="", flush=True)
        texts_q.task_done()
        time.sleep(PRINT_DURATION)


def audio_to_text(data: bytes) -> str:
    """ 音声データをテキストに変換

    Args:
        data (bytes): 音声データ

    Returns:
        str: 変換されたテキスト
    """
    recognizer = sr.Recognizer()

    loss_bytes = len(data) % (SAMPLE_RATE * 2)
    if loss_bytes != 0:
        data += b"\x00" * (SAMPLE_RATE * 2 - loss_bytes)

    recognized_text = ""
    try:
        audio_data = sr.AudioData(data, SAMPLE_RATE, 2)
        recognized_text = recognizer.recognize_google(
            audio_data, language="ja-JP",
            show_all=False)
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        pass
    finally:
        del recognizer
        return recognized_text


def recognition_process(data_q: Queue, texts_q: Queue) -> None:
    """ 音声データを受け取り、テキストに変換して出力用のキューに格納

    Args:
        data_q (Queue): 音声データを格納したキュー
        texts_q (Queue): 出力用のキュー
    """
    global recording_finish_flag
    
    while True:
        data = data_q.get()
        if data is None:
            data_q.task_done()
            break

        recognized_text = audio_to_text(data)
        if recognized_text == FINISH_TEXT:
            recording_finish_flag = True
            data_q.task_done()
            texts_q.put(None)
            break

        for c in recognized_text:
            texts_q.put(c)
        data_q.task_done()


def recording_process(data_q: Queue) -> None:
    """ 音声データを録音して音声データ用のキューに格納

    Args:
        data_q (Queue): 音声データを格納するキュー
    """
    global recording_finish_flag
    recording_finish_flag = False

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK_SIZE)
    
    low_count = 0
    voice_on = False
    data = b''

    stream.start_stream()
    while stream.is_active() and not recording_finish_flag:
        data_buffer = stream.read(CHUNK_SIZE)
        if np.max(np.abs(struct.unpack(f"{CHUNK_SIZE}h", data_buffer))) > AMP_THRESHOLD:
            if not voice_on:
                voice_on = True
            else:
                low_count = 0
            data += data_buffer
        elif voice_on:
            if low_count < LOW_COUNT_MAX:
                low_count += 1
                data += data_buffer
            else:
                low_count = 0
                voice_on = False
                data_q.put(data)
                data = b''
        
    stream.stop_stream()
    stream.close()
    audio.terminate()


def record_and_recognize() -> None:
    """ 録音と音声認識を行う """
    data_q = Queue()
    texts_q = Queue()

    recording_thread = threading.Thread(
        target=recording_process,
        args=(data_q,))
    recognition_thread = threading.Thread(
        target=recognition_process,
        args=(data_q, texts_q))
    print_thread = threading.Thread(
        target=print_gradually,
        args=(texts_q,))

    recording_thread.start()
    recognition_thread.start()
    print_thread.start()
    recording_thread.join()
    recognition_thread.join()
    data_q.join()
    print_thread.join()
    texts_q.join()


if __name__ == "__main__":
    list_audio_devices()
    record_and_recognize()