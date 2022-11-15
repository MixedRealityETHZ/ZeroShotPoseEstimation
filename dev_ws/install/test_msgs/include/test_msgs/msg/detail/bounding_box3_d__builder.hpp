// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from test_msgs:msg/BoundingBox3D.idl
// generated code does not contain a copyright notice

#ifndef TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__BUILDER_HPP_
#define TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__BUILDER_HPP_

#include "test_msgs/msg/detail/bounding_box3_d__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace test_msgs
{

namespace msg
{

namespace builder
{

class Init_BoundingBox3D_width
{
public:
  explicit Init_BoundingBox3D_width(::test_msgs::msg::BoundingBox3D & msg)
  : msg_(msg)
  {}
  ::test_msgs::msg::BoundingBox3D width(::test_msgs::msg::BoundingBox3D::_width_type arg)
  {
    msg_.width = std::move(arg);
    return std::move(msg_);
  }

private:
  ::test_msgs::msg::BoundingBox3D msg_;
};

class Init_BoundingBox3D_length
{
public:
  explicit Init_BoundingBox3D_length(::test_msgs::msg::BoundingBox3D & msg)
  : msg_(msg)
  {}
  Init_BoundingBox3D_width length(::test_msgs::msg::BoundingBox3D::_length_type arg)
  {
    msg_.length = std::move(arg);
    return Init_BoundingBox3D_width(msg_);
  }

private:
  ::test_msgs::msg::BoundingBox3D msg_;
};

class Init_BoundingBox3D_height
{
public:
  explicit Init_BoundingBox3D_height(::test_msgs::msg::BoundingBox3D & msg)
  : msg_(msg)
  {}
  Init_BoundingBox3D_length height(::test_msgs::msg::BoundingBox3D::_height_type arg)
  {
    msg_.height = std::move(arg);
    return Init_BoundingBox3D_length(msg_);
  }

private:
  ::test_msgs::msg::BoundingBox3D msg_;
};

class Init_BoundingBox3D_center
{
public:
  explicit Init_BoundingBox3D_center(::test_msgs::msg::BoundingBox3D & msg)
  : msg_(msg)
  {}
  Init_BoundingBox3D_height center(::test_msgs::msg::BoundingBox3D::_center_type arg)
  {
    msg_.center = std::move(arg);
    return Init_BoundingBox3D_height(msg_);
  }

private:
  ::test_msgs::msg::BoundingBox3D msg_;
};

class Init_BoundingBox3D_orientation
{
public:
  Init_BoundingBox3D_orientation()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_BoundingBox3D_center orientation(::test_msgs::msg::BoundingBox3D::_orientation_type arg)
  {
    msg_.orientation = std::move(arg);
    return Init_BoundingBox3D_center(msg_);
  }

private:
  ::test_msgs::msg::BoundingBox3D msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::test_msgs::msg::BoundingBox3D>()
{
  return test_msgs::msg::builder::Init_BoundingBox3D_orientation();
}

}  // namespace test_msgs

#endif  // TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__BUILDER_HPP_
