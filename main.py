import time
from Dicom_Reporter_Class import *

def main():
    input_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_database_phase_2'
    output_dir = r'Z:\Morfeus\Bastien\DICOM\EBRT_database_phase_2_nifti'

    supp_tags = {
    }

    contour_names = ['Bladder']
    contour_association = {
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
                                    nb_threads=32, verbose=True)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))


if __name__ == '__main__':
    main()
