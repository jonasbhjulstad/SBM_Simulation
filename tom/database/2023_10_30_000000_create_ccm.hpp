#pragma once

#include <tom/migration.hpp>

namespace Migrations
{

    struct CreateCCM : Migration
    {
        /*! Filename of the migration file. */
        T_MIGRATION

        /*! Run the migrations. */
        void up() const override
        {
            Schema::create("connection_community_map", [](Blueprint &table)
            {
                table.id();

                table.integer("p_out");
                table.integer("graph");
                table.integer("connection");
                table.integer("community");
                table.integer("weight");
            });
        }

        /*! Reverse the migrations. */
        void down() const override
        {
            Schema::dropIfExists("connection_community_map");
        }
    };

} // namespace Migrations
