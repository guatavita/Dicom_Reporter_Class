import os, glob
import threading
import time
import json
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from pathvalidate import sanitize_filepath
import numpy as np
import SimpleITK as sitk
import gdcm
import pydicom
import cv2

tags = {
    'SOPClassUID': '0008|0016',
    'StudyInstanceUID': '0020|000d',
    'StudyDescription': '0008|1030',
    'StudyDate': '0008|0020',
    'SeriesInstanceUID': '0020|000e',
    'SeriesDescription': '0008|103e',
    'SeriesDate': '0008|0021',
    'Modality': '0008|0060',
    'Manufacturer': '0008|0070',
    'InstitutionName': '0008|0080',
    'VolumeBasedCalculationTechnique': '0008|9207',
    'PresentationIntentType': '0008|0068',
    'PatientName': '0010|0010',
    'PatientID': '0010|0020',
    'PatientBirthDate': '0010|0030',
    'PatientAge': '0010|1010',
    'PatientSex': '0010|0040',
    'BodyPartExamined': '0018|0015',
    'ProtocolName': '0018|1030',
    'SliceThickness': '0018|0050',
    'SpacingBetweenSlices': '0018|0088',
    'DoseGridScaling': '3004|000e',
    'DoseSummationType': '3004|000a',
    'ROIContourSequence': '3006|0039',
    'StructureSetROISequence': '3006|0020',
}


def splitext_(path):
    if len(path.split('.')) > 2:
        return path.split('.')[0], '.'.join(path.split('.')[-2:])
    return os.path.splitext(path)


class AddDicomSeriesToDict(object):
    def __init__(self):
        self.series_reader = sitk.ImageSeriesReader()
        self.series_reader.SetGlobalWarningDisplay(False)
        self.series_reader.MetaDataDictionaryArrayUpdateOn()
        self.series_reader.LoadPrivateTagsOn()

        self.file_reader = sitk.ImageFileReader()
        self.file_reader.SetGlobalWarningDisplay(False)
        self.file_reader.LoadPrivateTagsOn()

    def run(self, dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict):
        series_ids_list = self.series_reader.GetGDCMSeriesIDs(dicom_folder)

        for series_id in series_ids_list:
            if series_id not in dicom_dict and series_id not in rd_dict:
                dicom_filenames = self.series_reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
                self.file_reader.SetFileName(dicom_filenames[0])
                try:
                    self.file_reader.Execute()
                except:
                    print('Reader failed on {} {}'.format(dicom_folder, series_id))
                    continue
                dictionary_creator(series_id, dicom_filenames, self.file_reader, dicom_dict, rd_dict, tags_dict)

        # this is to read RTSTRUCT
        rtstruct_files = glob.glob(os.path.join(dicom_folder, 'RS*.dcm'))
        if rtstruct_files:
            rtstruct_reader(rtstruct_files, rt_dict, tags_dict)


def dicom_reader_worker(A):
    q = A[0]
    while True:
        item = q.get()
        if item is None:
            break
        else:
            it, dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict, verbose = item
            dicom_series_to_dict = AddDicomSeriesToDict()
            if verbose:
                print("{}: {}".format(it, dicom_folder))
            try:
                dicom_series_to_dict.run(dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict)
            except:
                print('Failed on {}'.format(dicom_folder))
        q.task_done()


def return_series_ids(reader, input_folder, get_filenames=False):
    '''
    :param input_folder:
    :return: dictionary or list of the series ID per dicom
    '''
    series_ids_list = reader.GetGDCMSeriesIDs(input_folder)
    if get_filenames:
        series_dict = {}
        for series_id in series_ids_list:
            series_dict[series_id] = reader.GetGDCMSeriesFileNames(input_folder, series_id)
        return series_dict
    else:
        return series_ids_list


def dictionary_creator(series_id, dicom_filenames, reader, dicom_dict, rd_dict, tags_dict):
    '''
    :param series_id:
    :param dicom_filenames: list
    :param reader: sitk.ImageFileReader()
    :return:
    '''
    series_dict = {}
    series_dict['dicom_filenames'] = dicom_filenames
    if reader.HasMetaDataKey('0008|0060'):
        modality = reader.GetMetaData('0008|0060')
    else:
        modality = 'Unknown'

    for tag_name in list(tags_dict.keys()):
        tag_key = tags_dict.get(tag_name)
        if tag_key:
            if reader.HasMetaDataKey(tag_key):
                series_dict[tag_name] = reader.GetMetaData(tag_key)
            else:
                series_dict[tag_name] = None

    if modality.lower() == 'rtdose':
        if series_id not in rd_dict:
            rd_dict[series_id] = series_dict
    else:
        if series_id not in dicom_dict:
            dicom_dict[series_id] = series_dict


def rtstruct_reader(rtstruct_files, rt_dict, tags_dict):
    for rtstruct_file in rtstruct_files:
        try:
            ds = pydicom.read_file(rtstruct_file)
        except:
            print("Dicom cannot be read {}".format(rtstruct_file))
            return

        series_id = ds.get('SeriesInstanceUID')

        if series_id not in rt_dict:
            series_dict = {}
            series_dict['dicom_filenames'] = [rtstruct_file]
            for tag_name in list(tags_dict.keys()):
                tag_key = tags_dict.get(tag_name)
                if not tag_key:
                    continue
                series_dict[tag_name] = ds.get(tag_name)

            rt_dict[series_id] = series_dict


class Dicom_Reporter(object):
    def __init__(self, input_dir, output_dir=None, contour_names=[], contour_association={}, force_rewrite=False,
                 image_series_id=False, study_desc_name=True, merge_study_serie_desc=True, save_json=True,
                 load_json=True, supp_tags={}, nb_threads=int(0.5 * cpu_count()), verbose=False):
        '''
        :param input_dir: input folder where (unorganized) dicom can be found
        :param output_dir: output directory to save dcm_report.json and conversion output following \PatientID\StudyDate\StudyORSeriesDescription
        :param contour_names: list of contour names that will be written, ALL if empty
        :param contour_association: dictionary of contour names association
        :param force_rewrite: for rewrite of NIfTI images (user should remove dcm_report.json)
        :param image_series_id: True if you want the series id in the image filename, if you expect multiple series in study output dir
        :param study_desc_folder_name: True if you want the output folder to be named after the StudyDescription (False -> SeriesDescription)
        :param merge_study_serie_desc: merge study and series descript for image folder name
        :param save_json: save dcm_report.json in output_dir
        :param load_json: reload previous dcm_report.json
        :param supp_tags: extract DICOM metadata for in-house usage
        :param nb_threads: nb_thread to run processes in multithreads
        :param verbose: True to have output prints
        '''

        # TODO create dicom_report in excel sheet?
        # TODO remove series_id in output file name

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dicom_dict = {}
        self.rt_dict = {}
        self.rd_dict = {}
        self.contour_names = contour_names
        self.contour_association = contour_association
        self.force_rewrite = force_rewrite
        self.set_tags(supp_tags)
        self.nb_threads = min(nb_threads, int(0.9 * cpu_count()))
        self.image_series_id = image_series_id
        self.study_desc_name = study_desc_name
        self.merge_study_serie_desc = merge_study_serie_desc
        self.save_json = save_json
        self.load_json = load_json
        self.dcm_report_path = os.path.join(self.output_dir, 'dcm_report.json')
        self.verbose = verbose

        # folder that contains dicom
        self.folders_with_dcm = []

        # class init
        self.create_contour_association()
        self.load_dcm_report()
        self.walk_main_directory()
        self.dicom_explorer()
        self.create_association()
        self.save_dcm_report()

    def create_contour_association(self):
        if self.contour_names:
            for contour_name in self.contour_names:
                if not self.contour_association.get(contour_name):
                    self.contour_association[contour_name] = contour_name

            for contour_name in list(self.contour_association.keys()):
                self.contour_association[contour_name.lower()] = self.contour_association.get(contour_name)

    def create_association(self):

        if self.verbose:
            time_start = time.time()
            print("\nMerging RTDOSE:")
        # merging rdstruct to the corresponding study instance uid of the images
        for rd_series_key in list(self.rd_dict.keys()):
            rd_study_instance_uid = self.rd_dict[rd_series_key]['StudyInstanceUID']
            if not rd_study_instance_uid:
                continue
            for dcm_series_key in list(self.dicom_dict.keys()):
                dcm_study_instance_uid = self.dicom_dict[dcm_series_key]['StudyInstanceUID']
                if not dcm_study_instance_uid:
                    continue
                if dcm_study_instance_uid == rd_study_instance_uid:
                    if not self.dicom_dict[dcm_series_key].get('RTDOSE'):
                        self.dicom_dict[dcm_series_key]['RTDOSE'] = []
                    if rd_series_key not in self.dicom_dict[dcm_series_key]['RTDOSE']:
                        self.dicom_dict[dcm_series_key]['RTDOSE'].append(rd_series_key)

        if self.verbose:
            print("\nMerging RTSTRUCT:")
        # merging rtstruct to the corresponding study instance uid of the images
        for rt_series_key in list(self.rt_dict.keys()):
            rt_study_instance_uid = self.rt_dict[rt_series_key]['StudyInstanceUID']
            if not rt_study_instance_uid:
                continue
            for dcm_series_key in list(self.dicom_dict.keys()):
                dcm_study_instance_uid = self.dicom_dict[dcm_series_key]['StudyInstanceUID']
                if not dcm_study_instance_uid:
                    continue
                if dcm_study_instance_uid == rt_study_instance_uid:
                    if not self.dicom_dict[dcm_series_key].get('RTSTRUCT'):
                        self.dicom_dict[dcm_series_key]['RTSTRUCT'] = []
                    if rt_series_key not in self.dicom_dict[dcm_series_key]['RTSTRUCT']:
                        self.dicom_dict[dcm_series_key]['RTSTRUCT'].append(rt_series_key)
        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))

    def force_update(self):
        self.walk_main_directory()
        self.dicom_explorer()
        self.save_dcm_report()

    def load_dcm_report(self):
        if self.load_json:
            if os.path.exists(self.dcm_report_path):
                with open(self.dcm_report_path, 'r') as f:
                    self.dicom_dict = json.load(f)

    def save_dcm_report(self):
        if self.save_json and self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            with open(self.dcm_report_path, 'w') as f:
                json.dump(self.dicom_dict, f)

    def set_tags(self, supp_tags):
        try:
            if list(supp_tags.keys()):
                tags.update(supp_tags)
        except:
            raise ValueError("Provided supp_tags dict could not update initial dict.")

        self.tags_dict = tags

    def walk_main_directory(self):
        if self.verbose:
            time_start = time.time()
            print("\nLooking for DICOM:")
        for root, dirs, files in os.walk(self.input_dir, topdown=False):
            if glob.glob(os.path.join(root, '*.dcm')):
                self.folders_with_dcm.append(root)

        if self.verbose:
            print("A total of {} folders with DICOM files was found".format(len(self.folders_with_dcm)))
            print("Elapsed time {}s".format(int(time.time() - time_start)))

    def dicom_explorer(self):
        q = Queue(maxsize=self.nb_threads)
        A = (q,)
        threads = []
        for worker in range(self.nb_threads):
            t = Thread(target=dicom_reader_worker, args=(A,))
            t.start()
            threads.append(t)

        if self.verbose:
            time_start = time.time()
            print("\nReading DICOM:")
        for it, dicom_folder in enumerate(self.folders_with_dcm):
            item = [it, dicom_folder, self.dicom_dict, self.rd_dict, self.rt_dict, self.tags_dict, self.verbose]
            q.put(item)

        for worker in range(self.nb_threads):
            q.put(None)
        for t in threads:
            t.join()

        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))
            if self.dicom_dict:
                nb_dicom = 0
                for series_id in list(self.dicom_dict.keys()):
                    nb_dicom += len(self.dicom_dict[series_id]['dicom_filenames'])
                print("Nb DICOM files to process: {}".format(nb_dicom))

            if self.rd_dict:
                nb_rtdose = 0
                for series_id in list(self.rd_dict.keys()):
                    nb_rtdose += len(self.rd_dict[series_id]['dicom_filenames'])
                print("Nb RTDOSE files to process: {}".format(nb_rtdose))

            if self.rt_dict:
                nb_rtstruct = 0
                for series_id in list(self.rt_dict.keys()):
                    nb_rtstruct += len(self.rt_dict[series_id]['dicom_filenames'])
                print("Nb RTSTRUCT files to process: {}".format(nb_rtstruct))

    def rtdose_writer(self, output_dir, rtdose_series=[], dicom_handle=None):
        '''
        :param output_dir: output directory
        :param rtdose_series: list of RTDOSE id to point toward the dictionary in RTdose
        :param dicom_handle: reference dicom to resample RTDOSE if available
        :return:
        '''
        reader = sitk.ImageSeriesReader()
        reader.SetGlobalWarningDisplay(False)
        reader.MetaDataDictionaryArrayUpdateOff()
        reader.LoadPrivateTagsOff()
        i = 0
        for rtdose_series_id in rtdose_series:
            if not self.rd_dict.get(rtdose_series_id):
                continue
            reader.SetFileNames(self.rd_dict[rtdose_series_id]['dicom_filenames'])
            dose_handle = reader.Execute()
            origin = dose_handle.GetOrigin()
            spacing = dose_handle.GetSpacing()
            dose_array = np.squeeze(sitk.GetArrayFromImage(dose_handle).astype(np.float32))
            if self.rd_dict[rtdose_series_id]['DoseGridScaling']:
                dose_array = dose_array * float(self.rd_dict[rtdose_series_id]['DoseGridScaling'])
            if len(dose_array.shape) > 3:
                for c in range(dose_array.shape[0]):
                    dose_handle = sitk.GetImageFromArray(dose_array[c])
                    dose_handle.SetOrigin(origin[:3])
                    dose_handle.SetSpacing(spacing[:3])
                    if dicom_handle:
                        dose_handle = sitk.Resample(dose_handle, dicom_handle)
                    output_filename = os.path.join(output_dir, 'dose_{}_{}_{}.nii.gz'.format(i, self.rd_dict[
                        rtdose_series_id]['DoseSummationType'], c))
                    if self.force_rewrite or not os.path.exists(output_filename):
                        sitk.WriteImage(dose_handle, output_filename)
            else:
                dose_handle = sitk.GetImageFromArray(dose_array)
                dose_handle.SetOrigin(origin[:3])
                dose_handle.SetSpacing(spacing[:3])
                if dicom_handle:
                    dose_handle = sitk.Resample(dose_handle, dicom_handle)
                output_filename = os.path.join(output_dir, 'dose_{}_{}.nii.gz'.format(i, self.rd_dict[
                    rtdose_series_id]['DoseSummationType']))
                if self.force_rewrite or not os.path.exists(output_filename):
                    sitk.WriteImage(dose_handle, output_filename)
            i += 1

    def rtstruct_writer(self, output_dir, dicom_handle, rtstruct_series=[]):
        ref_size = dicom_handle.GetSize()
        ref_origin = dicom_handle.GetOrigin()
        ref_spacing = dicom_handle.GetSpacing()
        for rtstruct_series_id in rtstruct_series:
            if not self.rt_dict.get(rtstruct_series_id):
                continue
            try:
                for roi_structset, roi_contour in zip(self.rt_dict[rtstruct_series_id].get('StructureSetROISequence'),
                                                      self.rt_dict[rtstruct_series_id].get('ROIContourSequence')):
                    roi_name = roi_structset.ROIName
                    if self.contour_names and not self.contour_association.get(roi_name):
                        continue

                    if self.contour_association.get(roi_name):
                        roi_name = self.contour_association.get(roi_name)

                    output_filename = os.path.join(output_dir, '{}.nii.gz'.format(sanitize_filepath(roi_name)))
                    if self.force_rewrite or not os.path.exists(output_filename):
                        mask = np.zeros(ref_size[::-1], dtype=np.int8)
                        for contour_sequence in roi_contour.ContourSequence:
                            pts_list = [contour_sequence.ContourData[i:i + 3] for i in
                                        range(0, len(contour_sequence.ContourData), 3)]
                            pts_array = (np.array(pts_list) - ref_origin) / ref_spacing
                            slice_mask = cv2.fillPoly(np.zeros(ref_size[:2]), pts=[pts_array[:, :2].astype(np.int32)],
                                                      color=(255, 255, 255))
                            slice_id = int(pts_array[0, 2])
                            if np.any(mask[slice_id, :, :][slice_mask > 0]) == 1:
                                mask[slice_id, :, :][slice_mask > 0] = 0
                            else:
                                mask[slice_id, :, :][slice_mask > 0] = 1

                        mask_handle = sitk.GetImageFromArray(mask)
                        mask_handle.SetDirection(dicom_handle.GetDirection())
                        mask_handle.SetOrigin(ref_origin)
                        mask_handle.SetSpacing(ref_spacing)
                        sitk.WriteImage(mask_handle, output_filename)
            except:
                print('Failed to match rtstruct {}'.format(rtstruct_series_id))

    def dicom_writer_worker(self, A):
        q = A[0]
        while True:
            item = q.get()
            if item is None:
                break
            else:
                series_id, output_path = item
                reader = sitk.ImageSeriesReader()
                reader.SetGlobalWarningDisplay(False)
                reader.MetaDataDictionaryArrayUpdateOff()
                reader.LoadPrivateTagsOff()
                try:
                    if self.merge_study_serie_desc:
                        description = "{}_{}".format(
                            self.dicom_dict[series_id]['StudyDescription'].rstrip().replace(' ', '_'),
                            self.dicom_dict[series_id]['SeriesDescription'].rstrip().replace(' ', '_'))
                    else:
                        if self.study_desc_name:
                            description = self.dicom_dict[series_id]['StudyDescription'].rstrip().replace(' ', '_')
                        else:
                            description = self.dicom_dict[series_id]['SeriesDescription'].rstrip().replace(' ', '_')

                    description = ''.join(e for e in description if e.isalnum() or e == '_')
                    if self.dicom_dict[series_id]['PresentationIntentType']:
                        description += '_{}'.format(
                            self.dicom_dict[series_id]['PresentationIntentType'].replace(' ', '_'))
                    output_dir = os.path.join(output_path,
                                              '{}'.format(self.dicom_dict[series_id]['StudyDate']),
                                              '{}'.format(description))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if self.image_series_id:
                        output_filename = os.path.join(output_dir, 'image_series_{}.nii.gz'.format(series_id))
                    else:
                        output_filename = os.path.join(output_dir, 'image.nii.gz')

                    if self.force_rewrite or not os.path.exists(output_filename):
                        reader.SetFileNames(self.dicom_dict[series_id]['dicom_filenames'])
                        dicom_handle = reader.Execute()
                        identity_direction = tuple(np.identity(len(dicom_handle.GetSize())).flatten())
                        dicom_handle.SetDirection(identity_direction)
                        sitk.WriteImage(dicom_handle, output_filename)
                    else:
                        dicom_handle = sitk.ReadImage(output_filename)

                    if self.dicom_dict[series_id].get('RTDOSE'):
                        self.rtdose_writer(output_dir=output_dir, rtdose_series=self.dicom_dict[series_id]['RTDOSE'],
                                           dicom_handle=dicom_handle)

                    if self.dicom_dict[series_id].get('RTSTRUCT'):
                        self.rtstruct_writer(output_dir=output_dir, dicom_handle=dicom_handle,
                                             rtstruct_series=self.dicom_dict[series_id]['RTSTRUCT'])

                except:
                    print('Failed on {} {}'.format(series_id, output_path))
                q.task_done()

    def run_conversion(self):
        if not self.output_dir:
            raise ValueError("Output direction needs to be define (arg output_dir)")

        q = Queue(maxsize=self.nb_threads)
        A = (q,)
        threads = []
        for worker in range(self.nb_threads):
            t = Thread(target=self.dicom_writer_worker, args=(A,))
            t.start()
            threads.append(t)

        if self.verbose:
            time_start = time.time()
            print("\nConverting DICOM:")
        for series_id in list(self.dicom_dict.keys()):
            output_path = os.path.join(self.output_dir, self.dicom_dict[series_id]['PatientID'].rstrip())

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            item = [series_id, output_path]
            q.put(item)

        for worker in range(self.nb_threads):
            q.put(None)
        for t in threads:
            t.join()

        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))
