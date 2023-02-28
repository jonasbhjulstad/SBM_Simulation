    #ifndef SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP
    #define SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP
    #include <Sycl_Graph/Buffer/Base/Buffer.hpp>
    #include <Sycl_Graph/Graph/Invariant/Graph_Types.hpp>
    #include <tuple>
    namespace Sycl_Graph::Invariant
    {
    template <Sycl_Graph::Base::Buffer_type ... Bs>
    struct Buffer
    {
        typedef typename std::tuple_element_t<0, std::tuple<Bs ...>>::uI_t uI_t;
        typedef std::tuple<typename Bs::Container_t ...> Container_t;
        typedef std::tuple<typename Bs::Container_t::Data_t ...> Data_t;
        Buffer() = default;
        Buffer(Bs &&... buffers): buffers(buffers ...) {}
        Buffer(const std::tuple<Bs ...>& buffers): buffers(buffers) {}
        typedef Buffer<Bs ...> This_t;
        std::tuple<Bs ...> buffers;


        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
        
        template <typename T>
        struct Container_type
        {
            static constexpr bool value = (std::is_same_v<T, typename Bs::Container_t> || ...);
        };
        template <typename T>
        struct Container_Data_type
        {
            static constexpr bool value = (std::is_same_v<T, typename Bs::Container_t::Data_t> || ...);
        };
        template <typename C> requires Container_type<C>::value
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<C, typename Bs::Container_t ...>();
        }

        template <typename ... Cs> requires (Container_type<Cs>::value && ...)
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Cs)>{type_index<Cs>() ...};
        }

        template <typename C> requires Container_type<C>::value
        auto&& get_buffer()
        {
            //get index of buffer
            constexpr uI_t index = get_buffer_index<C>();
            return std::get<index>(buffers);
        }
        template <typename ... Cs> requires (Container_type<Cs>::value && ...)
        auto&& get_buffers()
        {
            return std::array{get_buffer<Cs>() ...};
        }

        template <typename D> requires Container_Data_type<D>::value
        static constexpr auto get_buffer_index()
        {
            return Sycl_Graph::index_of_type<Sycl_Graph::Base::Vertex<D, uI_t>, typename Bs::Container_t ...>();
        }

        template <typename ... Ds> requires (Container_Data_type<Ds>::value && ...)
        static constexpr auto get_buffer_index()
        {
            return std::array<uI_t, sizeof...(Ds)>{type_index<Sycl_Graph::Base::Vertex<Ds, uI_t>>() ...};
        }

        template <typename D> requires Container_Data_type<D>::value
        auto&& get_buffer()
        {
            constexpr uI_t index = get_buffer_index<D>();
            return std::get<index>(buffers);
        }
        template <typename ... Ds> requires (Container_Data_type<Ds>::value && ...)
        auto&& get_buffers()
        {
            return std::array{get_buffer<Ds>() ...};
        }

        auto size() const
        {
            return std::apply([](auto &&... buffers) {
                return (buffers.size() + ...);
            }, buffers);
        }

        template <typename C> requires Container_type<C>::value
        auto size() const
        {
            return get_buffer<C>().size();
        }

        template <typename ... Cs> requires (Container_type<Cs>::value && ...)
        void add(const std::vector<Cs> && ... vertices)
        {
            (get_buffer<Cs>().add(vertices), ...);
        }

        template <typename ... Ds> requires (Container_Data_type<Ds>::value && ...)
        void add(const std::vector<Ds>&& ... data)
        {
            (add(data), ...);
        }

        template <typename ... Cs> requires (Container_type<Cs>::value && ...)
        void remove(const std::vector<Cs>&&... elements)
        {
            ((get_buffer<Cs>().remove(elements), ...));
        }

        auto &operator=(This_t &&other)
        {
            buffers = std::move(other.buffers);
            return *this;
        }

        auto copy() const
        {
            Buffer B;
            B.buffers = this->buffers;
            return B;
        }

        auto &operator+(const This_t &other)
        {
            std::apply([&other](auto &&... buffers) {
                return std::make_tuple((buffers + other.buffers) ...);}, this->buffers);
            return *this;
        }
    };

    template <typename T>
    concept Buffer_type = Sycl_Graph::Base::Buffer_type<T>;
    } // namespace Sycl_Graph::Invariant

    #endif // SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP