// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from test_msgs:msg/BoundingBox3D.idl
// generated code does not contain a copyright notice

#ifndef TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_HPP_
#define TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"
// Member 'center'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__test_msgs__msg__BoundingBox3D __attribute__((deprecated))
#else
# define DEPRECATED__test_msgs__msg__BoundingBox3D __declspec(deprecated)
#endif

namespace test_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct BoundingBox3D_
{
  using Type = BoundingBox3D_<ContainerAllocator>;

  explicit BoundingBox3D_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : orientation(_init),
    center(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->height = 0.0;
      this->length = 0.0;
      this->width = 0.0;
    }
  }

  explicit BoundingBox3D_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : orientation(_alloc, _init),
    center(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->height = 0.0;
      this->length = 0.0;
      this->width = 0.0;
    }
  }

  // field types and members
  using _orientation_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _orientation_type orientation;
  using _center_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _center_type center;
  using _height_type =
    double;
  _height_type height;
  using _length_type =
    double;
  _length_type length;
  using _width_type =
    double;
  _width_type width;

  // setters for named parameter idiom
  Type & set__orientation(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->orientation = _arg;
    return *this;
  }
  Type & set__center(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->center = _arg;
    return *this;
  }
  Type & set__height(
    const double & _arg)
  {
    this->height = _arg;
    return *this;
  }
  Type & set__length(
    const double & _arg)
  {
    this->length = _arg;
    return *this;
  }
  Type & set__width(
    const double & _arg)
  {
    this->width = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    test_msgs::msg::BoundingBox3D_<ContainerAllocator> *;
  using ConstRawPtr =
    const test_msgs::msg::BoundingBox3D_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      test_msgs::msg::BoundingBox3D_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      test_msgs::msg::BoundingBox3D_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__test_msgs__msg__BoundingBox3D
    std::shared_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__test_msgs__msg__BoundingBox3D
    std::shared_ptr<test_msgs::msg::BoundingBox3D_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const BoundingBox3D_ & other) const
  {
    if (this->orientation != other.orientation) {
      return false;
    }
    if (this->center != other.center) {
      return false;
    }
    if (this->height != other.height) {
      return false;
    }
    if (this->length != other.length) {
      return false;
    }
    if (this->width != other.width) {
      return false;
    }
    return true;
  }
  bool operator!=(const BoundingBox3D_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct BoundingBox3D_

// alias to use template instance with default allocator
using BoundingBox3D =
  test_msgs::msg::BoundingBox3D_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace test_msgs

#endif  // TEST_MSGS__MSG__DETAIL__BOUNDING_BOX3_D__STRUCT_HPP_
