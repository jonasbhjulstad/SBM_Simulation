#include <orm/db.hpp>

#include <tom/application.hpp>

#include <2014_10_12_000000_create_posts_table.hpp>

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
    try {
        // Ownership of the shared_ptr()
        auto db = setupDatabaseManager();

        return TomApplication(argc, argv, std::move(db), "TOM_EXAMPLE_ENV")
                .migrations<CreatePostsTable>()
                .seeders<DatabaseSeeder>()
                // Fire it up 🔥🚀✨
                .run();

    } catch (const std::exception &e) {

        TomApplication::logException(e);
    }

    return EXIT_FAILURE;
}

std::shared_ptr<DatabaseManager> setupDatabaseManager()
{
    using namespace Orm::Constants; // NOLINT(google-build-using-namespace)

    // Ownership of the shared_ptr()
    return DB::create({
        {driver_,     QMYSQL},
        {host_,       qEnvironmentVariable("DB_MYSQL_HOST", H127001)},
        {port_,       qEnvironmentVariable("DB_MYSQL_PORT", P3306)},
        {database_,   qEnvironmentVariable("DB_MYSQL_DATABASE", EMPTY)},
        {username_,   qEnvironmentVariable("DB_MYSQL_USERNAME", EMPTY)},
        {password_,   qEnvironmentVariable("DB_MYSQL_PASSWORD", EMPTY)},
        {charset_,    qEnvironmentVariable("DB_MYSQL_CHARSET", UTF8MB4)},
        {collation_,  qEnvironmentVariable("DB_MYSQL_COLLATION", UTF8MB40900aici)},
        {timezone_,   TZ00},
        /* Specifies what time zone all QDateTime-s will have, the overridden default is
           the Qt::UTC, set to the Qt::LocalTime or QtTimeZoneType::DontConvert to use
           the system local time. */
        {qt_timezone, QVariant::fromValue(Qt::UTC)},
        {strict_,     true},
    },
        QStringLiteral("tinyorm_tom_mysql")); // shell:connection
}
