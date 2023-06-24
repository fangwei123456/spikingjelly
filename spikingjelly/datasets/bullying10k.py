from typing import Optional, Callable, List, Tuple
import os

from torch.utils.data import Dataset
from torchvision.datasets import utils


class Bullying10k(Dataset):

    def __init__(
        self, 
        root: str, 
        pose_estimation: bool = False,
        train: Optional[bool] = None,
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.pose_estimation = pose_estimation
        self.train = train
        self.data_type = data_type
        self.frames_number = frames_number
        self.split_by = split_by
        self.duration = duration
        self.custom_integrate_function = custom_integrate_function
        self.custom_integrated_frames_dir_name = custom_integrated_frames_dir_name
        self.transform = transform
        self.target_transform = target_transform

        download_root = os.path.join(root, 'download')
        extract_root = os.path.join(root, "extract")
        events_np_root = os.path.join(root, "events_np")
        resource_list = self.resource_url_md5()

        if not os.path.exists(events_np_root):
            os.removedirs(extract_root)

            if not os.path.exists(download_root):
                # download the dataset ()
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                if self.downloadable():
                    for file_name, url, md5 in resource_list:
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(
                        f'This dataset can not be downloaded by SpikingJelly, '
                        f'please download files manually and put files at [{download_root}]. '
                        f'The resources file_name, url, and md5 are: \n{resource_list}'
                    )
            else:
                # check the integrity of downloaded files
                print(
                    f"The [{download_root}] directory for saving downloaded "
                    f"files already exists, check files..."
                )
                for file_name, url, md5 in resource_list:
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')
                        if os.path.exists(fpath):
                            # If file is corrupted, remove it.
                            os.remove(fpath)
                            print(f'Remove [{fpath}]')
                        # download a new one
                        if self.downloadable():
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            raise NotImplementedError(
                                f"This dataset can not be downloaded by SpikingJelly, "
                                f"please download [{file_name}] from [{url}] manually "
                                f"and put files at {download_root}."
                            )

            # extract the files
            os.mkdir(extract_root)
            print(f'Mkdir [{extract_root}].')
            self.extract_downloaded_files(download_root, extract_root)

            # convert the original data to npz files
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
            print(
                f"Start to convert the origin data from [{extract_root}]"
                f" to [{events_np_root}] in np.ndarray format."
            )
            self.create_events_np_files(extract_root, events_np_root)

    @staticmethod
    def resource_url_md5() -> List[Tuple[str, str, str]]:
        return [
            ("handshake.zip", "https://figshare.com/ndownloader/files/41268834", "681d70f499e736a1e805305284ddc425"),
            ("slapping.zip", "https://figshare.com/ndownloader/files/41247021", "84b41d6805958f9f62f425223916ffc2"),
            ("punching.zip", "https://figshare.com/ndownloader/files/41263314", "40954f480ab210099d448b7b88fc4719"),
            ("walking.zip", "https://figshare.com/ndownloader/files/41247024", "56e4cac9c0814ce701c3b2292c15b6a9"),
            ("fingerguess.zip", "https://figshare.com/ndownloader/files/41253057", "f83114e5b4f0ea57cac86fb080c7e4d7"),
            ("strangling.zip", "https://figshare.com/ndownloader/files/41261904", "8185ecd6f3147e9b609d22f06270aa86"),
            ("greeting.zip", "https://figshare.com/ndownloader/files/41268792", "4a763fad728b04c8356db8544f1121fe"),
            ("pushing.zip", "https://figshare.com/ndownloader/files/41268951", "7986c74ade7149a98672120a89b13ba8"),
            ("hairgrabs.zip", "https://figshare.com/ndownloader/files/41277855", "a9cf690ed0a3305da4a4b8e110f64db1"),
            ("kicking.zip", "https://figshare.com/ndownloader/files/41278008", "6c3218f977de4ac29c84a10b17779c33"),
            ("val_keypoints.json", "https://figshare.com/ndownloader/files/41279925", "5cb7bc5c7e3500fe22bca2f17fe3dd0c"),
            ("train_keypoints.json", "https://figshare.com/ndownloader/files/41279946", "82e4ceef266f1347cf7142382d561293"),
        ]

    @staticmethod
    def downloadable() -> bool:
        return True

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str) -> None:
        pass

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str) -> None:
        pass
