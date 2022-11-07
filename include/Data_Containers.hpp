#ifndef DATA_CONTAINERS_HPP
#define DATA_CONTAINERS_HPP
#include <concepts>
#include <ranges>
#include <tuple>
#include <iterator>
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
concept GenericArray = ElementIterable<T> &&
Indexable<T> && requires(T x) {
  x.size(); // must have `x.size()`
};

template <typename T>
concept FixedSize = requires(T container)
                {
                    {std::tuple_size_v<T>}-> std::convertible_to<std::size_t>;
                };
template <typename T>
concept DynamicSize =
                std::ranges::random_access_range<T>
                && requires(T container)
                {
                    container.resize(0);
                    container.push_back({});
                };

template <typename T>
concept DynamicArray = GenericArray<T> && DynamicSize<T>;
template <typename T>
concept FixedArray = GenericArray<T> && FixedSize<T>;

} // namespace containers

#endif // DATA_CONTAINERS_HPP
