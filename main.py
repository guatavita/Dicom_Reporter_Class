from Dicom_Reporter_Class import *


def main():
    input_dir = r'C:\Bastien\DICOM_report_test\data'
    output_dir = os.path.join(input_dir, 'nifti')

    supp_tags = {
        'MammoDesc': '0055|1001'
    }

    contour_names = ['Bag_Bowel']
    contour_association = {
        'CTV45_Pelvis Nodes': 'CTVNs',
        'CTV45_Pelvic Nodes': 'CTVN',
        'CTV45_Pelvis_Nodes': 'CTVN',
        'CTVn45': 'CTVN',
        'ITV45_Uterus/Cervix/Vagina/Parametria': 'ITV45',
    }
    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=1, verbose=False)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))


if __name__ == '__main__':
    main()
