// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from test_msgs:msg/BoundingBox3D.idl
// generated code does not contain a copyright notice

#ifndef TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_H_
#define TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.h"
// Member 'center'
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/BoundingBox3D in the package test_msgs.
typedef struct test_msgs__msg__BoundingBox3D
{
  geometry_msgs__msg__Quaternion orientation;
  geometry_msgs__msg__Point center;
  double height;
  double length;
  double width;
} test_msgs__msg__BoundingBox3D;

// Struct for a sequence of test_msgs__msg__BoundingBox3D.
typedef struct test_msgs__msg__BoundingBox3D__Sequence
{
  test_msgs__msg__BoundingBox3D * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} test_msgs__msg__BoundingBox3D__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_H_
