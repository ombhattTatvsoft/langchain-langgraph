using System.ComponentModel.DataAnnotations;

namespace SlotBookingProject.Data;

public partial class Slot
{
    [Key]
    public int Id { get; set; }

    [StringLength(100)]
    public string? BookingName { get; set; }

    [DataType(DataType.Date)]
    public DateOnly? BookingDate { get; set; }

    public int? NoOfPeople { get; set; }

    [DataType(DataType.Time)]
    public TimeOnly? BookingTime { get; set; }

    [MinLength(10),MaxLength(13)]
    public string ContactNumber { get; set; } = null!;

    public bool IsActive { get; set; } 
} 
