import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_ACTIVITY = "next_activity"
  NEXT_TIME = "next_time"
  REMAINING_TIME = "remaining_time"

