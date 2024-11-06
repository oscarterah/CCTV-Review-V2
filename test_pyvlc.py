import vlc
import numpy as np
import cv2

class VideoPlayer:
    def __init__(self, video_path):
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media = self.instance.media_new_path(video_path)
        self.player.set_media(self.media)
        self.player.video_set_callbacks(self.lock, self.unlock, self.display)
        self.player.video_set_format("RV32", 640, 480, 640*4)
        self.frame = None

    def lock(self, *args):
        self.frame = np.empty((480, 640, 4), dtype=np.uint8)
        return self.frame.ctypes.data

    def unlock(self, *args):
        pass

    def display(self, *args):
        frame = self.frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    def play(self):
        self.player.play()

    def stop(self):
        self.player.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\oscar\Desktop\projects\test_labs\invasion.mp4"  # Change this to your video file path
    player = VideoPlayer(video_path)
    player.play()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        player.stop()
