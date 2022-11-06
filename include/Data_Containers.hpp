#ifndef DATA_CONTAINERS_HPP
#define DATA_CONTAINERS_HPP
#include <concepts>
#include <ranges>
#include <tuple>
namespace containers {
template <typename T>
concept Indexable = requires(T container) {
  container[0];
};
template <typename T>
concept ElementIterable = requires(std::ranges::range_value_t<T> x) {
  x.begin(); // must have `x.begin()`
  x.end();   // and `x.end()`
};

template <class T>
concept ResizableArray = Indexable<T> && ElementIterable<T> &&
    requires(T container) {
  container.resize(std::size_t{0});
};

template <class T>
concept FixedArray = Indexable<T> && ElementIterable<T> &&
    requires(T container) {
  std::tuple_size_v<T>->std::convertible_to<std::size_t>;
};
} // namespace containers

#endif // DATA_CONTAINERS_HPP
