"""Implementation of the 'completion', 'install-completion', and
'uninstall-completion' commands."""

from typing import Any


def cmd_install_completion(args: Any) -> int:
    """Handle the 'install-completion' command."""
    from xraylabtool.interfaces.completion_v2.integration import (
        install_completion_main,
    )

    return install_completion_main(args)


def cmd_uninstall_completion(args: Any) -> int:
    """Handle the 'uninstall-completion' command."""
    from xraylabtool.interfaces.completion_v2.integration import (
        uninstall_completion_main,
    )

    return uninstall_completion_main(args)


def cmd_completion(args: Any) -> int:
    """Handle the 'completion' command for the new completion system."""
    from xraylabtool.interfaces.completion_v2.installer import CompletionInstaller

    try:
        installer = CompletionInstaller()

        # Check if no action was specified
        if not hasattr(args, "completion_action") or args.completion_action is None:
            print(
                "❌ No action specified. Use 'xraylabtool completion --help' for usage information."
            )
            return 1

        if args.completion_action == "install":
            success = installer.install(
                shell=getattr(args, "shell", None),
                target_env=getattr(args, "env", None),
                force=getattr(args, "force", False),
            )
            return 0 if success else 1

        elif args.completion_action == "uninstall":
            success = installer.uninstall(
                target_env=getattr(args, "env", None),
                all_envs=getattr(args, "all", False),
            )
            return 0 if success else 1

        elif args.completion_action == "list":
            installer.list_environments()
            return 0

        elif args.completion_action == "status":
            installer.status()
            return 0

        elif args.completion_action == "info":
            from xraylabtool.interfaces.completion_v2.cli import show_completion_info

            show_completion_info()
            return 0

        else:
            print(f"❌ Unknown action: {args.completion_action}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
