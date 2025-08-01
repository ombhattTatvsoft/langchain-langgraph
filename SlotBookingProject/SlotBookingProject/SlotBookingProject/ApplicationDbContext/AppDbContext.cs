using Microsoft.EntityFrameworkCore;
using SlotBookingProject.Data;

namespace SlotBookingProject.ApplicationDbContext;

public partial class AppDbContext : DbContext
{
    public AppDbContext()
    {
    }

    public AppDbContext(DbContextOptions<AppDbContext> options)
        : base(options)
    {
    }

    public virtual DbSet<Slot> Slots { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Slot>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("Slots_pkey");

            entity.Property(e => e.Id).UseIdentityAlwaysColumn();
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}
