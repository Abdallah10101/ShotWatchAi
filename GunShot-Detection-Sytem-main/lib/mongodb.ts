// Dummy MongoDB helper for local/demo mode.
// The app no longer uses a database for authentication, but some modules
// still import `dbConnect`. To avoid runtime errors we provide a no-op here.

export async function dbConnect() {
  return null
}
