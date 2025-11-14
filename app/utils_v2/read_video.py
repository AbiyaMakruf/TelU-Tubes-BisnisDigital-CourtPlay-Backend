import cv2


def read_video(path_video, resize=False, width=1280, height=720):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()
    return frames, fps, original_width, original_height


def video_metadata(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frame_count


def frame_generator(path_video, resize=False, width=1280, height=720):
    cap = cv2.VideoCapture(path_video)
    idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, (width, height))
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class VideoFrameAccessor:
    def __init__(self, path_video, resize=False, width=1280, height=720):
        self.path_video = path_video
        self.resize = resize
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(path_video)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_index = -1
        self.current_frame = None

    def __len__(self):
        return self.length

    def _read_next(self):
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Frame index out of range")
        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))
        self.current_index += 1
        self.current_frame = frame
        return frame.copy()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.length)
            return [self[i] for i in range(start, stop, step)]
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError("Frame index out of range")
        if idx == self.current_index:
            return self.current_frame.copy()
        if idx < self.current_index or idx > self.current_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.current_index = idx - 1
        return self._read_next()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
