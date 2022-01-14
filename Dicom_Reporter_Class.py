import os, glob, copy
import threading
import time
import json
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from pathvalidate import sanitize_filepath
import numpy as np
import SimpleITK as sitk
import pydicom
import cv2
from tqdm import tqdm
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

tags = {
    'SOPClassUID': '0008|0016',
    'SOPInstanceUID': '0008|0018',
    'StudyInstanceUID': '0020|000d',
    'StudyDescription': '0008|1030',
    'StudyDate': '0008|0020',
    'SeriesInstanceUID': '0020|000e',
    'SeriesDescription': '0008|103e',
    'SeriesDate': '0008|0021',
    'FrameOfReferenceUID': '0020|0052',
    'ReferencedStudySequence': '0008|1110',
    'ReferencedFrameOfReferenceSequence': '3006|0010',
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


def AddDicomToDict(dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict):
    uid_dict = get_unique_uid_filenames(dicom_folder)
    for series_uid in list(uid_dict.keys()):
        # test only the series because rtstruct and rtdose will be separated by SOPInstanceUID later
        if series_uid not in dicom_dict:
            dicom_filenames = uid_dict.get(series_uid)
            dictionary_creator(series_uid, dicom_filenames, dicom_dict, rd_dict, rt_dict, tags_dict)


def get_unique_uid_filenames(dicom_folder):
    '''
    :param dicom_folder:
    :return: dictionary with list of .dcm files per SOPInstanceUID
    '''
    uid_filenames = {}
    filenames = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    for filename in filenames:
        try:
            ds = pydicom.dcmread(filename, stop_before_pixels=True)
        except:
            continue

        series_uid = ds.get('SeriesInstanceUID')
        if series_uid not in uid_filenames:
            uid_filenames[series_uid] = []

        # force test not None because you can have a SliceLocation == 0.0
        if ds.get('SliceLocation') is not None:
            slice_loc = float(ds.get('SliceLocation'))
        elif ds.get('InstanceNumber') is not None:
            slice_loc = float(ds.get('InstanceNumber'))
        else:
            slice_loc = 0
        uid_filenames[series_uid].append([slice_loc, filename])

    for series_uid in list(uid_filenames.keys()):
        uid_filenames.get(series_uid).sort()
        uid_filenames[series_uid] = [i[-1] for i in uid_filenames.get(series_uid)]

    return uid_filenames


def recurse(ds):
    # add extra tag that could be found only in sequences for some modalities
    # seq_elems = [elem for elem in ds if elem.VR == 'SQ']
    elem_list = []
    for key in list(ds.keys()):
        try:
            # have to do that to avoid "unable to convert value to int without loss" with ds.iterall()
            # previous version: {k.keyword:k.value for k in ds.iterall()}
            elem = ds.get(key)
        except:
            continue

        if elem.VR == 'SQ':
            elem_list += sum([recurse(item) for item in elem.value], [])
        else:
            elem_list += [elem]
    return elem_list


def dictionary_creator(series_uid, dicom_filenames, dicom_dict, rd_dict, rt_dict, tags_dict):
    '''
    :param series_uid:
    :param dicom_filenames: list
    :return:
    '''

    try:
        ds = pydicom.dcmread(dicom_filenames[0], stop_before_pixels=True)
    except:
        print("Dicom cannot be read {}".format(dicom_filenames[0]))
        return

    uid_dict = {}
    uid_dict['dicom_filenames'] = dicom_filenames
    modality = ds.get('Modality')
    ds_all = {k.keyword: k.value for k in recurse(ds)}
    for tag_name in list(tags_dict.keys()):
        if tag_name == 'PatientName':
            uid_dict[tag_name] = str(ds.get(tag_name))
        else:
            uid_dict[tag_name] = ds.get(tag_name)

        if not uid_dict[tag_name]:
            uid_dict[tag_name] = ds_all.get(tag_name)

    if modality.lower() == 'rtdose':
        if uid_dict['SOPInstanceUID'] not in rd_dict:
            rd_dict[uid_dict['SOPInstanceUID']] = uid_dict
    elif modality.lower() == 'rtstruct':
        if uid_dict['SOPInstanceUID'] not in rt_dict:
            rt_dict[uid_dict['SOPInstanceUID']] = uid_dict
    elif modality.lower() not in ['rtplan']:
        if series_uid not in dicom_dict:
            dicom_dict[series_uid] = uid_dict


def dicom_reader_worker(A):
    q = A[0]
    while True:
        item = q.get()
        if item is None:
            break
        else:
            dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict, verbose = item
            try:
                AddDicomToDict(dicom_folder, dicom_dict, rd_dict, rt_dict, tags_dict)
            except:
                print('Failed on {}'.format(dicom_folder))
        q.task_done()


def dicom_to_sitk(lstFilesDCM, force_uint16=False, force_int16=False):
    RefDs = pydicom.read_file(lstFilesDCM[0])
    if RefDs.get('NumberOfFrames'):
        ConstPixelDims = (int(RefDs.NumberOfFrames), int(RefDs.Rows), int(RefDs.Columns))
    else:
        ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    if len(lstFilesDCM) > 1:
        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[lstFilesDCM.index(filenameDCM), ...] = ds.pixel_array
    else:
        ArrayDicom[:, ...] = RefDs.pixel_array

    z_spacing = 1.0

    if len(lstFilesDCM) > 1:
        SdDs = pydicom.read_file(lstFilesDCM[1])
        if RefDs.get('SliceLocation') is not None and SdDs.get('SliceLocation') is not None:
            z_spacing = abs(RefDs.get('SliceLocation') - SdDs.get('SliceLocation'))
        elif RefDs.get('SliceThickness'):
            z_spacing = RefDs.get('SliceThickness')

    ArrayDicom = pydicom.pixel_data_handlers.apply_rescale(ArrayDicom, RefDs)

    if RefDs.PhotometricInterpretation == "MONOCHROME1":
        ArrayDicom = np.amax(ArrayDicom) - ArrayDicom

    if force_uint16:
        ArrayDicom = ArrayDicom.astype(np.uint16)

    if force_int16:
        ArrayDicom = ArrayDicom.astype(np.int16)

    if RefDs.get('PixelSpacing'):
        spacing = tuple(np.array(RefDs.PixelSpacing, dtype=np.float)) + (np.float(z_spacing),)
    else:
        spacing = (1.0, 1.0,) + (np.float(z_spacing),)

    # maybe use RefDs.ImageOrientationPatient
    identity_direction = tuple(np.identity(len(ConstPixelDims)).flatten())

    if RefDs.get('ImagePositionPatient'):
        origin = tuple(np.array(RefDs.ImagePositionPatient, dtype=np.float))
    else:
        origin = (0, 0, 0)
    sitk_pointer = sitk.GetImageFromArray(ArrayDicom)
    sitk_pointer.SetDirection(identity_direction)
    sitk_pointer.SetSpacing(spacing)
    sitk_pointer.SetOrigin(origin)
    return sitk_pointer


class Dicom_Reporter(object):
    def __init__(self, input_dir, output_dir=None, contour_names=[], contour_association={}, force_rewrite=False,
                 extension='.nii.gz', force_uint16=False, force_int16=False, image_series_id=False,
                 study_desc_name=True, merge_study_serie_desc=True,
                 avoid_duplicate=False, save_json=True, load_json=True, supp_tags={},
                 nb_threads=int(0.5 * cpu_count()), verbose=False):
        '''
        :param input_dir: input folder where (unorganized) dicom can be found
        :param output_dir: output directory to save dcm_report.json and conversion output following \PatientID\SeriesDate\StudyORSeriesDescription
        :param contour_names: list of contour names that will be written, ALL if empty
        :param contour_association: dictionary of contour names association
        :param force_rewrite: for rewrite of NIfTI images (user should remove dcm_report.json)
        :param extension: output image extension ['.nii.gz', '.nii', '.mhd', '.nrrd', '.mha']
        :param force_uint16: force_uint16 output pixel value representation
        :param force_int16: force_int16 output pixel value representation
        :param image_series_id: True if you want the series id in the image filename, if you expect multiple series in study output dir
        :param study_desc_folder_name: True if you want the output folder to be named after the StudyDescription (False -> SeriesDescription)
        :param merge_study_serie_desc: merge study and series descript for image folder name
        :param avoid_duplicate: True if you want to add _N after folder name in case duplicate output foldername
        :param save_json: save dcm_report.json in output_dir
        :param load_json: reload previous dcm_report.json
        :param supp_tags: extract DICOM metadata for in-house usage, format dict such as {'SOPClassUID': '0008|0016',}
        :param nb_threads: nb_thread to run processes in multithreads
        :param verbose: True to have output prints
        '''

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dicom_dict = {}
        self.rt_dict = {}
        self.rd_dict = {}
        self.contour_names = contour_names
        self.contour_association = contour_association
        self.force_rewrite = force_rewrite
        self.extension = extension
        self.force_uint16 = force_uint16
        self.force_int16 = force_int16
        self.set_tags(supp_tags)
        self.nb_threads = min(nb_threads, int(0.9 * cpu_count()))
        self.image_series_id = image_series_id
        self.study_desc_name = study_desc_name
        self.merge_study_serie_desc = merge_study_serie_desc
        self.avoid_duplicate = avoid_duplicate
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
        self.convert_types()
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
        for rd_sop_uid_key in list(self.rd_dict.keys()):
            rd_study_instance_uid = self.rd_dict[rd_sop_uid_key]['StudyInstanceUID']
            rd_frame_reference = self.rd_dict[rd_sop_uid_key]['FrameOfReferenceUID']
            rd_reference_uid = None
            if self.rd_dict[rd_sop_uid_key]['ReferencedStudySequence']:
                rd_reference_uid = self.rd_dict[rd_sop_uid_key]['ReferencedStudySequence'][0].get(
                    'ReferencedSOPInstanceUID')
            if None in [rd_study_instance_uid, rd_reference_uid, rd_frame_reference]:
                continue
            for dcm_series_uid_key in list(self.dicom_dict.keys()):
                dcm_study_instance_uid = self.dicom_dict[dcm_series_uid_key]['StudyInstanceUID']
                dcm_frame_reference = self.dicom_dict[dcm_series_uid_key]['FrameOfReferenceUID']
                if None in [dcm_study_instance_uid, dcm_frame_reference]:
                    continue
                if all([dcm_study_instance_uid == rd_study_instance_uid, rd_reference_uid == dcm_study_instance_uid,
                        rd_frame_reference == dcm_frame_reference]):
                    if not self.dicom_dict[dcm_series_uid_key].get('RTDOSE'):
                        self.dicom_dict[dcm_series_uid_key]['RTDOSE'] = []
                    if rd_sop_uid_key not in self.dicom_dict[dcm_series_uid_key]['RTDOSE']:
                        self.dicom_dict[dcm_series_uid_key]['RTDOSE'].append(rd_sop_uid_key)

        if self.verbose:
            print("\nMerging RTSTRUCT:")
        # merging rtstruct to the corresponding study instance uid of the images
        for rt_sop_uid_key in list(self.rt_dict.keys()):
            rt_study_instance_uid = self.rt_dict[rt_sop_uid_key]['StudyInstanceUID']
            rt_frame_reference = self.rt_dict[rt_sop_uid_key]['FrameOfReferenceUID']
            rt_reference_uid = None
            if self.rt_dict[rt_sop_uid_key]['ReferencedFrameOfReferenceSequence']:
                rt_reference_uid = \
                self.rt_dict[rt_sop_uid_key]['ReferencedFrameOfReferenceSequence'][0].get('RTReferencedStudySequence')[
                    0].get('ReferencedSOPInstanceUID')
            if None in [rt_study_instance_uid, rt_reference_uid, rt_frame_reference]:
                continue
            for dcm_series_uid_key in list(self.dicom_dict.keys()):
                dcm_study_instance_uid = self.dicom_dict[dcm_series_uid_key]['StudyInstanceUID']
                dcm_frame_reference = self.dicom_dict[dcm_series_uid_key]['FrameOfReferenceUID']
                if None in [dcm_study_instance_uid, dcm_frame_reference]:
                    continue
                if all([dcm_study_instance_uid == rt_study_instance_uid, rt_reference_uid == dcm_study_instance_uid,
                        rt_frame_reference == dcm_frame_reference]):
                    if not self.dicom_dict[dcm_series_uid_key].get('RTSTRUCT'):
                        self.dicom_dict[dcm_series_uid_key]['RTSTRUCT'] = []
                    if rt_sop_uid_key not in self.dicom_dict[dcm_series_uid_key]['RTSTRUCT']:
                        self.dicom_dict[dcm_series_uid_key]['RTSTRUCT'].append(rt_sop_uid_key)
        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))

    def convert_types(self):
        for dcm_uid_key in list(self.dicom_dict.keys()):
            for tag in list(self.dicom_dict[dcm_uid_key].keys()):
                if isinstance(self.dicom_dict[dcm_uid_key][tag], pydicom.multival.MultiValue):
                    self.dicom_dict[dcm_uid_key][tag] = list(self.dicom_dict[dcm_uid_key][tag])

    def force_update(self):
        self.walk_main_directory()
        self.dicom_explorer()
        self.save_dcm_report()

    def load_dcm_report(self):
        if self.load_json:
            if os.path.exists(self.dcm_report_path):
                with open(self.dcm_report_path, 'r') as f:
                    try:
                        self.dicom_dict = json.load(f)
                    except:
                        print("JSON file could not be loaded")

    def save_dcm_report(self):
        if self.save_json and self.output_dir:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            with open(self.dcm_report_path, 'w') as f:
                try:
                    json.dump(self.dicom_dict, f)
                except:
                    print("JSON file could not be saved")

            # number of keys per element of the dict should be same because we define the header of the txt file
            keys = list(self.dicom_dict.keys())
            tags = list(self.dicom_dict[keys[0]].keys())
            output_txt = open(self.dcm_report_path.replace('.json', '.txt'), 'w')
            output_txt.write('{}\n'.format(','.join(tags)))
            for key in keys:
                for tag in tags:
                    tag_value = self.dicom_dict[key].get(tag)
                    if isinstance(tag_value, str):
                        tag_value = tag_value.replace(',', '')
                    if isinstance(tag_value, list):
                        tag_value = '/'.join(str(x) for x in tag_value)
                    output_txt.write('{},'.format(tag_value))
                output_txt.write('\n')
            output_txt.close()

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
        for dicom_folder in tqdm(self.folders_with_dcm):
            item = [dicom_folder, self.dicom_dict, self.rd_dict, self.rt_dict, self.tags_dict, self.verbose]
            q.put(item)

        for worker in range(self.nb_threads):
            q.put(None)
        for t in threads:
            t.join()

        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))
            if self.dicom_dict:
                nb_dicom = 0
                for dicom_uid in list(self.dicom_dict.keys()):
                    nb_dicom += len(self.dicom_dict[dicom_uid]['dicom_filenames'])
                print("Nb DICOM files to process: {}".format(nb_dicom))

            if self.rd_dict:
                nb_rtdose = 0
                for sop_uid in list(self.rd_dict.keys()):
                    nb_rtdose += len(self.rd_dict[sop_uid]['dicom_filenames'])
                print("Nb RTDOSE files to process: {}".format(nb_rtdose))

            if self.rt_dict:
                nb_rtstruct = 0
                for sop_uid in list(self.rt_dict.keys()):
                    nb_rtstruct += len(self.rt_dict[sop_uid]['dicom_filenames'])
                print("Nb RTSTRUCT files to process: {}".format(nb_rtstruct))

    def rtdose_writer(self, output_dir, rtdose_sop_uid_list=[], dicom_handle=None):
        '''
        :param output_dir: output directory
        :param rtdose_sop_uid_list: list of RTDOSE id to point toward in the dictionary
        :param dicom_handle: reference dicom to resample RTDOSE if available
        :return:
        '''

        i = 0
        for rtdose_sop_uid in rtdose_sop_uid_list:
            if not self.rd_dict.get(rtdose_sop_uid):
                continue
            dose_handle = dicom_to_sitk(self.rd_dict[rtdose_sop_uid]['dicom_filenames'], force_uint16=False,
                                        force_int16=False)
            origin = dose_handle.GetOrigin()
            spacing = dose_handle.GetSpacing()
            dose_array = np.squeeze(sitk.GetArrayFromImage(dose_handle).astype(np.float32))
            if self.rd_dict[rtdose_sop_uid]['DoseGridScaling']:
                dose_array = dose_array * float(self.rd_dict[rtdose_sop_uid]['DoseGridScaling'])
            if len(dose_array.shape) > 3:
                for c in range(dose_array.shape[0]):
                    dose_handle = sitk.GetImageFromArray(dose_array[c])
                    dose_handle.SetOrigin(origin[:3])
                    dose_handle.SetSpacing(spacing[:3])
                    if dicom_handle:
                        dose_handle = sitk.Resample(dose_handle, dicom_handle)
                    output_filename = os.path.join(output_dir, 'dose_{}_{}_{}{}'.format(i, self.rd_dict[
                        rtdose_sop_uid]['DoseSummationType'], c, self.extension))
                    if self.force_rewrite or not os.path.exists(output_filename):
                        sitk.WriteImage(dose_handle, output_filename)
            else:
                dose_handle = sitk.GetImageFromArray(dose_array)
                dose_handle.SetOrigin(origin[:3])
                dose_handle.SetSpacing(spacing[:3])
                if dicom_handle:
                    dose_handle = sitk.Resample(dose_handle, dicom_handle)
                output_filename = os.path.join(output_dir, 'dose_{}_{}{}'.format(i, self.rd_dict[
                    rtdose_sop_uid]['DoseSummationType'], self.extension))
                if self.force_rewrite or not os.path.exists(output_filename):
                    sitk.WriteImage(dose_handle, output_filename)
            i += 1

    def rtstruct_writer(self, output_dir, dicom_handle, rtstruct_sop_uid_list=[]):
        ref_size = dicom_handle.GetSize()
        ref_origin = dicom_handle.GetOrigin()
        ref_spacing = dicom_handle.GetSpacing()
        for rtstruct_sop_uid in rtstruct_sop_uid_list:
            if not self.rt_dict.get(rtstruct_sop_uid):
                continue
            try:
                for roi_structset, roi_contour in zip(self.rt_dict[rtstruct_sop_uid].get('StructureSetROISequence'),
                                                      self.rt_dict[rtstruct_sop_uid].get('ROIContourSequence')):
                    roi_name = roi_structset.ROIName
                    if self.contour_names and not self.contour_association.get(roi_name):
                        continue

                    if roi_contour.get('ContourSequence') is None:
                        continue

                    if self.contour_association.get(roi_name):
                        roi_name = self.contour_association.get(roi_name)

                    if roi_contour.ContourSequence[0].ContourGeometricType.lower() == 'point':
                        output_filename = os.path.join(output_dir,
                                                       'point_{}{}'.format(sanitize_filepath(roi_name), self.extension))
                    else:
                        output_filename = os.path.join(output_dir,
                                                       '{}{}'.format(sanitize_filepath(roi_name), self.extension))

                    if self.force_rewrite or not os.path.exists(output_filename):
                        mask = np.zeros(ref_size[::-1], dtype=np.int8)
                        for contour_sequence in roi_contour.ContourSequence:
                            pts_list = [contour_sequence.ContourData[i:i + 3] for i in
                                        range(0, len(contour_sequence.ContourData), 3)]
                            # pts_array = (np.array(pts_list) - ref_origin) / ref_spacing
                            pts_array = np.array(
                                [dicom_handle.TransformPhysicalPointToIndex(i) for i in np.array(pts_list)])
                            slice_mask = cv2.fillPoly(np.zeros(ref_size[:2]), [pts_array[:, :2].astype(np.int32)], 1)
                            slice_id = int(pts_array[0, 2])
                            mask[slice_id, :, :][slice_mask > 0] += 1

                        mask = mask % 2
                        mask_handle = sitk.GetImageFromArray(mask)
                        mask_handle.SetDirection(dicom_handle.GetDirection())
                        mask_handle.SetOrigin(ref_origin)
                        mask_handle.SetSpacing(ref_spacing)
                        sitk.WriteImage(mask_handle, output_filename)
            except:
                print('Failed to match rtstruct {} out {}'.format(rtstruct_sop_uid, output_dir))

    def dicom_writer_worker(self, A):
        q = A[0]
        while True:
            item = q.get()
            if item is None:
                break
            else:
                dcm_uid, output_dir = item
                try:
                    if self.image_series_id:
                        output_filename = os.path.join(output_dir,
                                                       'image_series_{}{}'.format(dcm_uid, self.extension))
                    else:
                        output_filename = os.path.join(output_dir, 'image{}'.format(self.extension))

                    if self.force_rewrite or not os.path.exists(output_filename):
                        dicom_handle = dicom_to_sitk(self.dicom_dict[dcm_uid]['dicom_filenames'],
                                                     force_uint16=self.force_uint16, force_int16=self.force_int16)
                        sitk.WriteImage(dicom_handle, output_filename)
                    else:
                        # not sure if that is multithread safe either
                        dicom_handle = sitk.ReadImage(output_filename)

                    if self.dicom_dict[dcm_uid].get('RTDOSE'):
                        self.rtdose_writer(output_dir=output_dir,
                                           rtdose_sop_uid_list=self.dicom_dict[dcm_uid]['RTDOSE'],
                                           dicom_handle=dicom_handle)

                    if self.dicom_dict[dcm_uid].get('RTSTRUCT'):
                        self.rtstruct_writer(output_dir=output_dir, dicom_handle=dicom_handle,
                                             rtstruct_sop_uid_list=self.dicom_dict[dcm_uid]['RTSTRUCT'])
                except:
                    print('Failed on {} {}'.format(dcm_uid, output_dir))
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
        for dcm_uid in tqdm(list(self.dicom_dict.keys())):
            output_path = os.path.join(self.output_dir, self.dicom_dict[dcm_uid]['PatientID'].rstrip())

            if self.merge_study_serie_desc:
                description = "{}_{}".format(
                    str(self.dicom_dict[dcm_uid]['StudyDescription']).rstrip().replace(' ', '_'),
                    str(self.dicom_dict[dcm_uid]['SeriesDescription']).rstrip().replace(' ', '_'))
            else:
                if self.study_desc_name:
                    description = str(self.dicom_dict[dcm_uid]['StudyDescription']).rstrip().replace(' ', '_')
                else:
                    description = str(self.dicom_dict[dcm_uid]['SeriesDescription']).rstrip().replace(' ', '_')

            description = ''.join(e for e in description if e.isalnum() or e == '_')
            if self.dicom_dict[dcm_uid]['PresentationIntentType']:
                description += '_{}'.format(
                    self.dicom_dict[dcm_uid]['PresentationIntentType'].replace(' ', '_'))
            output_dir = os.path.join(output_path,
                                      '{}'.format(self.dicom_dict[dcm_uid]['StudyDate']),
                                      '{}'.format(description))

            # avoid duplicate folder name if no descriptions are available
            if os.path.exists(output_dir) and self.avoid_duplicate:
                last_output_dir = dir_list = glob.glob(output_dir+'*')[-1]
                if last_output_dir[-1].isdigit():
                    output_dir = last_output_dir[:-1] + str(int(last_output_dir[-1]) + 1)
                else:
                    output_dir = output_dir + '_1'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            item = [dcm_uid, output_dir]
            q.put(item)

        for worker in range(self.nb_threads):
            q.put(None)
        for t in threads:
            t.join()

        if self.verbose:
            print("Elapsed time {}s".format(int(time.time() - time_start)))
