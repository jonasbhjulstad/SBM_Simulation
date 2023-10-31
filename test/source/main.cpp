
#define _GLIBCXX_USE_CXX11_ABI 0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <orm/db.hpp>
#include <doctest/doctest.h>
#include <filesystem>
static const std::string cwd = std::filesystem::current_path().generic_string();

// Ownership of a shared_ptr()
auto manager = Orm::DB::create({
    {"driver",                  "QSQLITE"},
    {"database",                qEnvironmentVariable("DB_DATABASE", "./HelloWorld.sqlite3")},
    {"foreign_key_constraints", qEnvironmentVariable("DB_FOREIGN_KEYS", "true")},
    {"check_database_exists",   false},
    /* Specifies what time zone all QDateTime-s will have, the overridden default is
       the Qt::UTC, set to the Qt::LocalTime or QtTimeZoneType::DontConvert to use
       the system local time. */
    {"qt_timezone",             QVariant::fromValue(Qt::UTC)},
    /* Return a QDateTime with the correct time zone instead of the QString,
       only works when the qt_timezone isn't set to the DontConvert. */
    {"return_qdatetime",        true},
    {"prefix",                  ""},
    {"prefix_indexes",          false},
});
