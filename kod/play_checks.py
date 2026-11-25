import math


def validate_scenario_commands(mapping):
    for name, command in mapping.items():
        if not isinstance(name, str) or len(command) != 3:
            raise ValueError(f"Geçersiz senaryo: {name}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in command):
            raise ValueError(f"Komut sonlu sayılardan oluşmalı: {name}")
        if any(abs(value) > 1.0 for value in command):
            raise ValueError(f"Komut eğitim aralığı dışında (-1,1): {name} {command}")


def resume_plan(target_iterations: int, checkpoint_iteration: int | None) -> tuple[int, int]:
    """Son checkpoint'ten sonraki iterasyonu ve kalan sayıyı döndürür."""
    if target_iterations <= 0:
        raise ValueError("Hedef iterasyon sayısı pozitif olmalı")
    if checkpoint_iteration is None:
        return 0, target_iterations
    if checkpoint_iteration < 0:
        raise ValueError("Checkpoint iterasyonu negatif olamaz")
    start = checkpoint_iteration + 1
    remaining = target_iterations - start
    if remaining <= 0:
        raise ValueError(
            f"Checkpoint {checkpoint_iteration}. iterasyonu içeriyor ({start} güncelleme tamam); hedef toplam "
            f"{target_iterations}. Daha uzun eğitmek için --max_iterations ile daha büyük bir TOPLAM hedef ver."
        )
    return start, remaining
