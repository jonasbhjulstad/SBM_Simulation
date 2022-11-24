template <template <typename> typename T, typename D>
struct Foo
{
    T<D> data;
};

int main()