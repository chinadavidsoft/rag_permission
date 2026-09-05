import argparse

from rag_permission.auth import create_access_token
from rag_permission.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local demo JWT")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--ttl-minutes", type=int, default=None)
    args = parser.parse_args()
    settings = Settings()
    groups = tuple(group.strip() for group in args.groups.split(",") if group.strip())
    print(
        create_access_token(
            args.user_id,
            groups,
            settings.auth_secret,
            args.ttl_minutes or settings.auth_token_ttl_minutes,
        )
    )


if __name__ == "__main__":
    main()
