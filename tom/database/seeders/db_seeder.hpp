#pragma once

#include <tom/seeder.hpp>

namespace Seeders
{

    /*! Main database seeder. */
    struct DatabaseSeeder : Seeder
    {
        /*! Run the database seeders. */
        void run() override
        {
            DB::table("posts")->insert({
                {{"name", "1. post"}},
                {{"name", "2. post"}},
            });
        }
    };

} // namespace Seeders
