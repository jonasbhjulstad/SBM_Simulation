#include <soci/soci.h>
#include <vector>
using namespace soci;
// Example 3.
void fill_ids(std::vector<int> &ids)
{
    for (std::size_t i = 0; i < ids.size(); ++i)
        ids[i] = i; // mimics source of a new ID
}
int main()
{
    const int BATCH_SIZE = 25;
    std::vector<int> ids(BATCH_SIZE);
    soci::session sql("postgresql", "user=postgres password=postgres");

    statement st = (sql.prepare << "insert into numbers(value) values(:val)", use(ids));
    for (int i = 0; i != 4; ++i)
    {
        fill_ids(ids);
        st.execute(true);
    }
}
