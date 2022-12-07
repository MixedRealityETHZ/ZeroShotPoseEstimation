std_msgs/Header header      # Header timestamp should be acquisition time of image, frame_id is object ID
geometry_msgs/Pose pose     # 6D object pose

float64 width               # Bounding Box x_size, meters
float64 height              # Bounding Box y_size, meters
float64 length              # Bounding Box z_size, meters