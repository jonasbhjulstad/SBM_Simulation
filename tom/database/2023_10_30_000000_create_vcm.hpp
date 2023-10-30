#pragma once

#include <tom/migration.hpp>

namespace Migrations
{

    struct CreateVCM : Migration
    {
        /*! Filename of the migration file. */
        T_MIGRATION

        /*! Run the migrations. */
        void up() const override
        {
            Schema::create("vertex_community_map", [](Blueprint &table)
            {
                table.id();

                table.integer("p_out");
                table.integer("graph");
                table.integer("vertex");
                table.integer("community");
            });
        }

        /*! Reverse the migrations. */
        void down() const override
        {
            Schema::dropIfExists("vertex_community_map");
        }
    };

} // namespace Migrations
