// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from test_msgs:msg/BoundingBox3D.idl
// generated code does not contain a copyright notice

#ifndef TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__TRAITS_HPP_
#define TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__TRAITS_HPP_

#include "test_msgs/msg/detail/bounding_box3_d__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"
// Member 'center'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<test_msgs::msg::BoundingBox3D>()
{
  return "test_msgs::msg::BoundingBox3D";
}

template<>
inline const char * name<test_msgs::msg::BoundingBox3D>()
{
  return "test_msgs/msg/BoundingBox3D";
}

template<>
struct has_fixed_size<test_msgs::msg::BoundingBox3D>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point>::value && has_fixed_size<geometry_msgs::msg::Quaternion>::value> {};

template<>
struct has_bounded_size<test_msgs::msg::BoundingBox3D>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point>::value && has_bounded_size<geometry_msgs::msg::Quaternion>::value> {};

template<>
struct is_message<test_msgs::msg::BoundingBox3D>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__TRAITS_HPP_
