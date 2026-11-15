package floats

import "os"

// onCI reports a GitHub Actions run.
func onCI() bool { return os.Getenv("GITHUB_ACTIONS") != "" }
