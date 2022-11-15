// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from test_msgs:msg/BoundingBox3D.idl
// generated code does not contain a copyright notice
#include "test_msgs/msg/detail/bounding_box3_d__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `orientation`
#include "geometry_msgs/msg/detail/quaternion__functions.h"
// Member `center`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
test_msgs__msg__BoundingBox3D__init(test_msgs__msg__BoundingBox3D * msg)
{
  if (!msg) {
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__init(&msg->orientation)) {
    test_msgs__msg__BoundingBox3D__fini(msg);
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point__init(&msg->center)) {
    test_msgs__msg__BoundingBox3D__fini(msg);
    return false;
  }
  // height
  // length
  // width
  return true;
}

void
test_msgs__msg__BoundingBox3D__fini(test_msgs__msg__BoundingBox3D * msg)
{
  if (!msg) {
    return;
  }
  // orientation
  geometry_msgs__msg__Quaternion__fini(&msg->orientation);
  // center
  geometry_msgs__msg__Point__fini(&msg->center);
  // height
  // length
  // width
}

bool
test_msgs__msg__BoundingBox3D__are_equal(const test_msgs__msg__BoundingBox3D * lhs, const test_msgs__msg__BoundingBox3D * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->orientation), &(rhs->orientation)))
  {
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->center), &(rhs->center)))
  {
    return false;
  }
  // height
  if (lhs->height != rhs->height) {
    return false;
  }
  // length
  if (lhs->length != rhs->length) {
    return false;
  }
  // width
  if (lhs->width != rhs->width) {
    return false;
  }
  return true;
}

bool
test_msgs__msg__BoundingBox3D__copy(
  const test_msgs__msg__BoundingBox3D * input,
  test_msgs__msg__BoundingBox3D * output)
{
  if (!input || !output) {
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->orientation), &(output->orientation)))
  {
    return false;
  }
  // center
  if (!geometry_msgs__msg__Point__copy(
      &(input->center), &(output->center)))
  {
    return false;
  }
  // height
  output->height = input->height;
  // length
  output->length = input->length;
  // width
  output->width = input->width;
  return true;
}

test_msgs__msg__BoundingBox3D *
test_msgs__msg__BoundingBox3D__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  test_msgs__msg__BoundingBox3D * msg = (test_msgs__msg__BoundingBox3D *)allocator.allocate(sizeof(test_msgs__msg__BoundingBox3D), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(test_msgs__msg__BoundingBox3D));
  bool success = test_msgs__msg__BoundingBox3D__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
test_msgs__msg__BoundingBox3D__destroy(test_msgs__msg__BoundingBox3D * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    test_msgs__msg__BoundingBox3D__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
test_msgs__msg__BoundingBox3D__Sequence__init(test_msgs__msg__BoundingBox3D__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  test_msgs__msg__BoundingBox3D * data = NULL;

  if (size) {
    data = (test_msgs__msg__BoundingBox3D *)allocator.zero_allocate(size, sizeof(test_msgs__msg__BoundingBox3D), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = test_msgs__msg__BoundingBox3D__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        test_msgs__msg__BoundingBox3D__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
test_msgs__msg__BoundingBox3D__Sequence__fini(test_msgs__msg__BoundingBox3D__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      test_msgs__msg__BoundingBox3D__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

test_msgs__msg__BoundingBox3D__Sequence *
test_msgs__msg__BoundingBox3D__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  test_msgs__msg__BoundingBox3D__Sequence * array = (test_msgs__msg__BoundingBox3D__Sequence *)allocator.allocate(sizeof(test_msgs__msg__BoundingBox3D__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = test_msgs__msg__BoundingBox3D__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
test_msgs__msg__BoundingBox3D__Sequence__destroy(test_msgs__msg__BoundingBox3D__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    test_msgs__msg__BoundingBox3D__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
test_msgs__msg__BoundingBox3D__Sequence__are_equal(const test_msgs__msg__BoundingBox3D__Sequence * lhs, const test_msgs__msg__BoundingBox3D__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!test_msgs__msg__BoundingBox3D__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
test_msgs__msg__BoundingBox3D__Sequence__copy(
  const test_msgs__msg__BoundingBox3D__Sequence * input,
  test_msgs__msg__BoundingBox3D__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(test_msgs__msg__BoundingBox3D);
    test_msgs__msg__BoundingBox3D * data =
      (test_msgs__msg__BoundingBox3D *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!test_msgs__msg__BoundingBox3D__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          test_msgs__msg__BoundingBox3D__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!test_msgs__msg__BoundingBox3D__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
