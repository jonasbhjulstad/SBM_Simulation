#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <orm/db.hpp>
#include <string>

void create_hello_db(auto cwd) {
  std::system(
      ("sqlite3 " + cwd + "/HelloWorld.sqlite3 'create table if not exists posts(id INTEGER NOT NULL PRIMARY KEY "
      "AUTOINCREMENT, name VARCHAR NOT NULL); insert into posts values(1, 'First Post'); insert "
      "into posts values(2, 'Second Post'); select * from posts;'").c_str());
}

void remove_hello_db(auto cwd) { std::system(("rm -rf" + cwd + "HelloWorld.sqlite3").c_str()); }

void orm_hello() {
  using Orm::DB;
  auto cwd = std::filesystem::current_path().generic_string();
  remove_hello_db(cwd);
  create_hello_db(cwd);
  // Ownership of a shared_ptr()
  // Ownership of a shared_ptr()
  DB::statement("CREATE table IF NOT EXISTS test_table(num INTEGER);");
  remove_hello_db(cwd);
}

TEST_CASE("orm_hello") { orm_hello(); }
