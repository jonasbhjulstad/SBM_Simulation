#pragma once

#include <tom/migration.hpp>

namespace Migrations
{

    struct CreatePostsTable : Migration
    {
        /*! Filename of the migration file. */
        T_MIGRATION

        /*! Run the migrations. */
        void up() const override
        {
            Schema::create("posts", [](Blueprint &table)
            {
                table.id();

                table.string(NAME);
                table.timestamps();
            });
        }

        /*! Reverse the migrations. */
        void down() const override
        {
            Schema::dropIfExists("posts");
        }
    };

} // namespace Migrations
