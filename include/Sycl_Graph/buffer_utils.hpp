
void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter)
{
    for (const auto &t_i_i : state_iter)
    {
        file << t_i_i;
        if (&t_i_i != &state_iter.back())
            file << ",";
        else
            file << "\n";
    }
}

void linewrite(std::ofstream &file, const std::vector<float> &val)
{
    for (const auto &t_i_i : val)
    {
        file << t_i_i;
        if (&t_i_i != &val.back())
            file << ",";
        else
            file << "\n";
    }
}

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter)
{
    for (const auto &t_i : state_iter)
    {
        for (const auto &t_i_i : t_i)
        {
            file << t_i_i;
            file << ",";
        }
    }
    file << "\n";
}

template <typename T>
sycl::buffer<T, 1> buffer_create_1D(sycl::queue &q, const std::vector<T> &data, sycl::event &res_event)
{
    sycl::buffer<T> tmp(data.data(), data.size());
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return result;
}

template <typename T>
sycl::buffer<T, 2> buffer_create_2D(sycl::queue &q, const std::vector<std::vector<T>> &data, sycl::event &res_event)
{
    assert(std::all_of(data.begin(), data.end(), [&](const auto subdata)
                       { return subdata.size() == data[0].size(); }));

    std::vector<T> data_flat(data.size() * data[0].size());
    for (uint32_t i = 0; i < data.size(); ++i)
    {
        std::copy(data[i].begin(), data[i].end(), data_flat.begin() + i * data[0].size());
    }

    sycl::buffer<T, 2> tmp(data_flat.data(), sycl::range<2>(data.size(), data[0].size()));
    sycl::buffer<T, 2> result(sycl::range<2>(data.size(), data[0].size()));
    res_event = q.submit([&](sycl::handler &h)
                         {
        auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
        auto res_acc = result.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(result.get_range(), [=](sycl::id<2> idx)
                       { res_acc[idx] = tmp_acc[idx]; }); });

    return result;
}
sycl::buffer<uint32_t> generate_seeds(sycl::queue &q, uint32_t N_rng,
                                      uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 1000000);
    std::vector<uint32_t> rngs(N_rng);
    std::generate(rngs.begin(), rngs.end(), [&]()
                  { return dis(gen); });

    sycl::buffer<uint32_t> tmp(rngs.data(), rngs.size());
    sycl::buffer<uint32_t> result(sycl::range<1>(rngs.size()));

    q.submit([&](sycl::handler &h)
             {
        auto tmp_acc = tmp.get_access<sycl::access::mode::read>(h);
        auto res_acc = result.get_access<sycl::access::mode::write>(h);

        h.parallel_for(result.get_range(), [=](sycl::id<1> idx)
                       { res_acc[idx] = tmp_acc[idx]; }); });

    return result;
}


template <typename T>
std::vector<std::vector<T>> read_buffer(sycl::queue &q, sycl::buffer<T, 2> &buf,
                                        auto events = {})
{

    auto range = buf.get_range();
    auto rows = range[0];
    auto cols = range[1];

    std::vector<T> data(cols * rows);
    T *p_data = data.data();

    q.submit([&](sycl::handler &h)
             {
        //create accessor
        h.depends_on(events);
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, p_data); })
        .wait();

    // transform to 2D vector
    std::vector<std::vector<T>> data_2d(rows);
    for (int i = 0; i < rows; i++)
    {
        data_2d[i] = std::vector<T>(cols);
        for (int j = 0; j < cols; j++)
        {
            data_2d[i][j] = data[i * cols + j];
        }
    }

    return data_2d;
}


template <typename T>
std::vector<std::vector<T>> diff(const std::vector<std::vector<T>> &v)
{
    std::vector<std::vector<T>> res(v.size() - 1, std::vector<T>(v[0].size()));
    for (int i = 0; i < v.size() - 1; i++)
    {
        for (int j = 0; j < v[i].size(); j++)
        {
            res[i][j] = v[i + 1][j] - v[i][j];
        }
    }
    return res;
}
