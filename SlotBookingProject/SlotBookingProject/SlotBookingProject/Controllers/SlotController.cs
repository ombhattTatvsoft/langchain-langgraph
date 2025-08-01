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
            List<Slot> slots = await _context.Slots.Where(s => s.BookingDate == date && s.IsActive ==  true).ToListAsync();
            return Ok(slots);
        }

        [HttpPost]
        public async Task<IActionResult> AddSlot(Slot slot)
        {
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

        [HttpGet("GetSlotByContactAndDate")]
        public async Task<IActionResult> GetSlotByContactNumber(string contactNumber, DateOnly bookingDate)
        {
            List<Slot> slots = await _context.Slots.Where(s => s.ContactNumber.Contains(contactNumber) && s.BookingDate == bookingDate && s.IsActive == true).ToListAsync();
            return Ok(slots);
        }
    }
}
