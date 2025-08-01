using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace SlotBookingProject.Migrations
{
    /// <inheritdoc />
    public partial class dbcreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Slots",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityAlwaysColumn),
                    BookingName = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    BookingDate = table.Column<DateOnly>(type: "date", nullable: true),
                    NoOfPeople = table.Column<int>(type: "integer", nullable: true),
                    BookingTime = table.Column<TimeOnly>(type: "time without time zone", nullable: true),
                    ContactNumber = table.Column<string>(type: "character varying(13)", maxLength: 13, nullable: false),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("Slots_pkey", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Slots");
        }
    }
}
