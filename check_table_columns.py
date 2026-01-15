# Connection details - use postgres container directly with SSH tunnel
# For the columns query which doesn't require external connection
import subprocess
import sys

result = subprocess.run([
    'docker', 'exec', '-i', 'supabase-db', 
    'psql', '-U', 'postgres', '-d', 'postgres', '-c',
    """
    SELECT column_name, data_type, is_nullable 
    FROM information_schema.columns 
    WHERE table_name IN ('incidents', 'incident_tickets')
    ORDER BY table_name, ordinal_position
    """
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    sys.exit(1)

# Parse output
print(result.stdout)
