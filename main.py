from Dicom_Reporter_Class import Dicom_Reporter


def main():
    input_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_database_phase_2'
    output_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_database_phase_2_nifti'

    supp_tags = {
    }

    contour_names = ['Cervix-Uterus', 'Bladder', 'Rectum', 'Sigmoid', 'Vagina', 'Parametrium', 'Femur_R', 'Femur_L',
                     'Kidney_R', 'Kidney_L', 'Spinal', 'Bowel']
    contour_association = {
    }
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=1, verbose=True)
    dicom_explorer.run_conversion()


if __name__ == '__main__':
    main()
