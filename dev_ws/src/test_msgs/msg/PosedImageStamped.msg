std_msgs/Header header  # Header timestamp should be acquisition time of image

geometry_msgs/Pose pose # 6D camera pose at acquisition time
float64[9] intrinsics   # Intrinsic camera matrix for the raw images: [fx 0 cx 0 fy cy 0 0 1]

uint32 height           # image height, that is, number of rows
uint32 width            # image width, that is, number of columns

string encoding         # image encoding (e.g. RGB)
uint32 step             # values per pixel (e.g. step=3 if RGB)
uint8[] data            # actual matrix data, size is (step * (rows * columns))