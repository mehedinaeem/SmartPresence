from src.video.frame_sampler import sampled_frames


def process_video(video_path: str, callback, mode: str = "experiment") -> int:
    count = 0
    for frame_index, timestamp, frame in sampled_frames(video_path, mode):
        callback(frame, frame_index, timestamp); count += 1
    return count

