#pragma once

#include <tom/migration.hpp>

namespace Migrations
{

    struct CreateECM : Migration
    {
        /*! Filename of the migration file. */
        T_MIGRATION

        /*! Run the migrations. */
        void up() const override
        {
            Schema::create("edge_connection_map", [](Blueprint &table)
            {
                table.id();

                table.integer("p_out");
                table.integer("graph");
                table.integer("edge");
                table.integer("connection");
            });
        }

        /*! Reverse the migrations. */
        void down() const override
        {
            Schema::dropIfExists("edge_connection_map");
        }
    };

} // namespace Migrations
