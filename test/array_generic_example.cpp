template <typename D0, typename D1, template <typename, int> typename FixedArray_t>
struct Foo
{
    FixedArray_t<D0, 10> a;
    FixedArray_t<D1, 5> b;
};

template <typename D0, typename D1, template <typename> typename DynamicArray_t>
struct Bar
{
    Bar(int N0, int N1): a(N0), b(N1) {}
    DynamicArray_t<D0> a;
    DynamicArray_t<D1> b;
};


template <typename Container>
struct ContainerHolder
{
    
    Container c;
};