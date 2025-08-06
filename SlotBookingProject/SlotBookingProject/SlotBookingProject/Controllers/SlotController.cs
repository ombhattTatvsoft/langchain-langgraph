using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SlotBookingProject.ApplicationDbContext;
using SlotBookingProject.Data;

namespace SlotBookingProject.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SlotController : ControllerBase
    {
        private readonly AppDbContext _context;

        public SlotController(AppDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public async Task<IActionResult> GetSlots(DateOnly date)
        {
            List<Slot> slots = await _context.Slots.Where(s => s.BookingDate == date && s.IsActive == true).ToListAsync();
            return Ok(slots);
        }

        [HttpPost]
        public async Task<IActionResult> AddSlot(Slot slot)
        {
            // var fromTime = slot.BookingTime!.Value.AddMinutes(-59);
            // var toTime = slot.BookingTime!.Value.AddMinutes(59);

            // bool conflictExists = await _context.Slots.AnyAsync(s =>
            //     s.BookingDate!.Value == slot.BookingDate &&     
            //     s.BookingTime >= fromTime && s.BookingTime <= toTime
            // );

            // if (conflictExists)
            // {
            //     return Conflict("A slot already exists within ±1 hour of the selected time.");
            // }

            _context.Slots.Add(slot);
            await _context.SaveChangesAsync();
            return Ok(slot);
        }

        [HttpPatch]
        public async Task<IActionResult> CancelSlot(int slotId)
        {
            Slot? slot = await _context.Slots.FirstOrDefaultAsync(s => s.Id == slotId);
            if (slot != null)
            {
                slot.IsActive = false;
                await _context.SaveChangesAsync();
            }
            return Ok(slot);
        }

        [HttpPut]
        public async Task<IActionResult> UpdateSlot(Slot slot)
        {
            _context.Slots.Update(slot);
            await _context.SaveChangesAsync();
            return Ok(slot);
        }

        [HttpGet("GetSlotByContact")]
        public async Task<IActionResult> GetSlotByContactNumber(string contactNumber)
        {
            List<Slot> slots = await _context.Slots.Where(s => s.ContactNumber.Contains(contactNumber) && s.IsActive == true).ToListAsync();
            return Ok(slots);
        }
    }
}
