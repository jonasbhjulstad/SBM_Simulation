#include <orm/db.hpp>

#include <tom/application.hpp>
#include "database/migration_headers.hpp"
#include "tom_config.hpp"
#include <seeders/db_seeder.hpp>

using Orm::DatabaseManager;
using Orm::DB;

using TomApplication = Tom::Application;

using namespace Migrations; // NOLINT(google-build-using-namespace)
using namespace Seeders;    // NOLINT(google-build-using-namespace)

/*! Create the database manager instance and add a database connection. */
std::shared_ptr<DatabaseManager> setupDatabaseManager();

/*! C++ main function. */
int main(int argc, char *argv[])
{
    try
    {
        // Ownership of the shared_ptr()
        auto db = setupDatabaseManager();

        return TomApplication(argc, argv, std::move(db), "TOM_EXAMPLE_ENV")
            .migrations<CreateCCM, CreateECM,CreateEdgelists, CreateVCM>()
            .seeders<DatabaseSeeder>()
            // Fire it up 🔥🚀✨
            .run();
    }
    catch (const std::exception &e)
    {

        TomApplication::logException(e);
    }

    return EXIT_FAILURE;
}

std::shared_ptr<DatabaseManager> setupDatabaseManager()
{
    using namespace Orm::Constants; // NOLINT(google-build-using-namespace)

    // Ownership of the shared_ptr()
    return DB::create({
        {"driver", tom_config::TOM_DB_DRIVER},
        {"database", qEnvironmentVariable("DB_DATABASE", tom_config::SQLITE3_FILENAME)},
        {"foreign_key_constraints", qEnvironmentVariable("DB_FOREIGN_KEYS", "true")},
        {"check_database_exists", false},
        /* Specifies what time zone all QDateTime-s will have, the overridden default is
           the Qt::UTC, set to the Qt::LocalTime or QtTimeZoneType::DontConvert to use
           the system local time. */
        {"qt_timezone", QVariant::fromValue(Qt::UTC)},
        /* Return a QDateTime with the correct time zone instead of the QString,
           only works when the qt_timezone isn't set to the DontConvert. */
        {"return_qdatetime", true},
        {"prefix", ""},
        {"prefix_indexes", false},
    });
}
