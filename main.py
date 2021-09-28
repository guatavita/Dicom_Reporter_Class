from Dicom_Reporter_Class import Dicom_Reporter


def main():
    input_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_test'
    output_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_test\nifti'

    supp_tags = {
    }

    contour_names = []
    contour_association = {
    }
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    contour_names=contour_names,
                                    contour_association=contour_association,
                                    force_rewrite=True,
                                    image_series_id=False, study_desc_name=True,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=1, verbose=True)
    dicom_explorer.run_conversion()


if __name__ == '__main__':
    main()
