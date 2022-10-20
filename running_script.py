import subprocess

def run_command(command):
    ret = subprocess.run(command, capture_output=True, shell=True)
    print(ret.stdout.decode())

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj before_nms --where_apply_calib_class before_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj before_nms --where_apply_calib_class after_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj before_nms --where_apply_calib_class None --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj after_nms --where_apply_calib_class before_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj after_nms --where_apply_calib_class after_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj after_nms --where_apply_calib_class None --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj None --where_apply_calib_class before_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj None --where_apply_calib_class after_nms --calibrator 'isotonic'"
run_command(command)

command = "python calibrate.py --config 'config/opt/opt_VOC_detect_S_Hyp0.yaml' --where_apply_calib_obj None --where_apply_calib_class None --calibrator 'isotonic'"
run_command(command)
