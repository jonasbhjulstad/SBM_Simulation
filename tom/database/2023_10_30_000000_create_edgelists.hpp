#pragma once

#include <tom/migration.hpp>

namespace Migrations
{

    struct CreateEdgelists : Migration
    {
        /*! Filename of the migration file. */
        T_MIGRATION

        /*! Run the migrations. */
        void up() const override
        {
            Schema::create("edgelists", [](Blueprint &table)
            {
                table.id();

                table.integer("p_out");
                table.integer("graph");
                table.integer("from");
                table.integer("to");
            });
        }

        /*! Reverse the migrations. */
        void down() const override
        {
            Schema::dropIfExists("edgelists");
        }
    };

} // namespace Migrations
