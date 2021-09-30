import os
import glob
from queue import Queue
from threading import Thread
from multiprocessing import cpu_count
import SimpleITK as sitk


class Dicom_Reader(object):
    def __init__(self, input_dir, nb_threads=int(0.9 * cpu_count())):
        self.input_dir = input_dir
        self.folders_with_dcm = []
        self.nb_threads = nb_threads
        self.walk_root_folder()
        self.read_dicom_series()

    def walk_root_folder(self):
        for root, dirs, files in os.walk(self.input_dir, topdown=False):
            if glob.glob(os.path.join(root, '*.dcm')):
                self.folders_with_dcm.append(root)

    def dicom_reader_worker(self, A):
        q = A[0]
        while True:
            item = q.get()
            if item is None:
                break
            else:
                it, dicom_folder = item
                reader = sitk.ImageSeriesReader()
                reader.SetGlobalWarningDisplay(False)
                print('running {} {}'.format(it, dicom_folder))
                try:
                    series_ids_list = reader.GetGDCMSeriesIDs(dicom_folder)
                except:
                    print('Failed on {}'.format(dicom_folder))
            q.task_done()

    def read_dicom_series(self):
        q = Queue(maxsize=self.nb_threads)
        A = (q,)
        threads = []
        for worker in range(self.nb_threads):
            t = Thread(target=self.dicom_reader_worker, args=(A,))
            t.start()
            threads.append(t)

        for it, dicom_folder in enumerate(self.folders_with_dcm):
            item = [it, dicom_folder]
            q.put(item)

        for worker in range(self.nb_threads):
            q.put(None)
        for t in threads:
            t.join()


def main():
    input_dir = r'/workspace/Morfeus/Bastien/MERIT/DICOM_study_v1/Consolidated/DATA'
    dicom_explorer = Dicom_Reader(input_dir=input_dir)


if __name__ == '__main__':
    main()
