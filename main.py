from Dicom_Reporter_Class import *


def main():
    root = r'Z:\Morfeus'
    input_dir = os.path.join(root, r'Bastien\MERIT\DICOM_explore_example')
    output_dir = os.path.join(input_dir, 'niftii')
    supp_tags = {
        'MammoDesc': '0055|1001'
    }

    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    force_rewrite=True,
                                    save_json=True,
                                    load_json=False,
                                    supp_tags=supp_tags,
                                    nb_threads=32, verbose=False)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))

    time_start = time.time()
    dicom_explorer = Dicom_Reporter(input_dir=input_dir,
                                    output_dir=output_dir,
                                    force_rewrite=False,
                                    save_json=True,
                                    load_json=True,
                                    supp_tags=supp_tags,
                                    nb_threads=32, verbose=False)
    dicom_explorer.run_conversion()
    print("     Elapse time {}".format(time.time() - time_start))
    # dicom_explorer.dicom_dict


if __name__ == '__main__':
    main()
