from spoon_bot.bus.events import OutboundMessage
from spoon_bot.channels.delivery import (
    ChannelDeliveryService,
    DeliveryBinding,
    binding_from_session_key,
    conversation_scope_from_binding,
)
from spoon_bot.session.manager import Session


def test_delivery_service_resolves_explicit_account_binding():
    service = ChannelDeliveryService()

    binding = service.resolve_binding(
        explicit={
            "channel": "telegram",
            "account_id": "main_bot",
            "target": {"chat_id": "12345"},
        }
    )

    assert binding is not None
    assert binding.channel == "telegram:main_bot"
    assert binding.target["chat_id"] == "12345"


async def _deliver_with(service: ChannelDeliveryService, binding: DeliveryBinding) -> OutboundMessage:
    return await service.deliver("hello", binding)


def test_delivery_service_uses_session_metadata_before_session_key_fallback():
    service = ChannelDeliveryService()
    session = Session(
        session_key="telegram_main_bot_12345",
        metadata={
            "delivery_binding": {
                "channel": "discord:ops_bot",
                "target": {"channel_id": "999"},
            }
        },
    )

    binding = service.resolve_binding(session=session, session_key=session.session_key)

    assert binding is not None
    assert binding.channel == "discord:ops_bot"
    assert binding.target["channel_id"] == "999"


def test_session_key_fallback_parses_structured_channel_keys():
    binding = binding_from_session_key("discord_prod_bot_321")

    assert binding is not None
    assert binding.channel == "discord:prod_bot"
    assert binding.target["channel_id"] == "321"


def test_delivery_binding_metadata_derives_missing_account_id_from_channel():
    binding = DeliveryBinding.from_metadata(
        {
            "channel": "telegram:spoon_bot",
            "target": {"chat_id": "12345"},
        }
    )

    assert binding is not None
    assert binding.account_id == "spoon_bot"


def test_conversation_scope_from_binding_uses_stable_chat_scope():
    scope = conversation_scope_from_binding(
        DeliveryBinding(
            channel="telegram:spoon_bot",
            account_id="spoon_bot",
            session_key="telegram_spoon_bot_12345",
            target={"chat_id": "12345"},
        )
    )

    assert scope is not None
    assert scope.channel == "telegram"
    assert scope.account_id == "spoon_bot"
    assert scope.conversation_id == "12345"


def test_delivery_service_deliver_falls_back_from_legacy_channel_name():
    sent: list[OutboundMessage] = []

    class FakeChannel:
        async def send(self, message: OutboundMessage) -> None:
            sent.append(message)

    channels = {"telegram:spoon_bot": FakeChannel()}
    service = ChannelDeliveryService(
        channel_lookup=lambda name: channels.get(name),
        channel_names_lookup=lambda: list(channels.keys()),
    )

    import asyncio

    asyncio.run(
        _deliver_with(
            service,
            DeliveryBinding(
                channel="telegram",
                account_id="spoon_bot",
                target={"chat_id": "12345"},
            ),
        )
    )

    assert len(sent) == 1
    assert sent[0].channel == "telegram:spoon_bot"
    assert sent[0].metadata["chat_id"] == "12345"
