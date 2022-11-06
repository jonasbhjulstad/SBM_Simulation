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
concept ElementIterable = requires(T x) {
  x.begin(); // must have `x.begin()`
  x.end();   // and `x.end()`
};

template <typename T>
concept FixedArray = std::ranges::contiguous_range<std::ranges::range_value_t<T>> &&
                std::ranges::sized_range<T> &&
                std::ranges::random_access_range<T> &&
                std::ranges::contiguous_range<T>
                && requires(T container)
                {
                    {std::tuple_size_v<T>}-> std::convertible_to<std::size_t>;
                };
template <typename T>
concept DynamicArray =
                std::ranges::random_access_range<T>
                && requires(T container)
                {
                    container.resize(0);
                };


} // namespace containers

#endif // DATA_CONTAINERS_HPP
