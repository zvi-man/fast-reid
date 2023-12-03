from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class VehicleHoldingTest:
    FRAME_COL_NAME = 'frame'
    TRACK_ID_COL_NAME = 'track_id'
    SCORE_COL_NAME = 'score'

    @classmethod
    def plot_vehicles_scores_as_time(cls, df: pd.DataFrame, threshold: float = 0.5) -> None:
        min_frame = int(df[cls.FRAME_COL_NAME].min())
        max_frame = int(df[cls.FRAME_COL_NAME].max())
        frame_vec = np.array(list(range(min_frame, max_frame+1)))
        all_track_ids = df[cls.TRACK_ID_COL_NAME].unique()
        for track_id in all_track_ids:
            df_track_id = df[df[cls.TRACK_ID_COL_NAME] == track_id]
            track_frames = list(df_track_id[cls.FRAME_COL_NAME])
            track_scores = list(df_track_id[cls.SCORE_COL_NAME])
            plt.plot(track_frames, track_scores, label=f"Track {track_id}")

            # track_id_scores = np.array([0.0] * len(frame_vec))
            # track_frame_indexes = [list(frame_vec).index(frame) for frame in track_frames]
            # track_id_scores[track_frame_indexes] = list(df_track_id[cls.SCORE_COL_NAME])
            # plt.plot(frame_vec, track_id_scores, label=f"Track {track_id}")

        plt.plot(frame_vec, [threshold] * len(frame_vec), label=f"Threshold", linestyle=":")
        plt.legend()
        plt.show()


    @classmethod
    def get_detection_time(cls, df: pd.DataFrame, gt_tracklet: str, threshold: int) -> Union[int, float]:
        pass

    @classmethod
    def get_precision(cls, df: pd.DataFrame, gt_tracklet: str, threshold: int) -> float:
        pass

    # TODO: add function that receives many dfs and calculates the final graph for
    #  detection time and precision as function of threshold


if __name__ == '__main__':
    df_single_gt = pd.read_csv("data.csv")
    VehicleHoldingTest.plot_vehicles_scores_as_time(df_single_gt)
