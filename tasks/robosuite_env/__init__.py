def get_env(env_name, ranges, **kwargs):
    if env_name == 'SawyerPickPlaceDistractor':
        from robosuite_env.tasks.pick_place import SawyerPickPlaceDistractor
        env = SawyerPickPlaceDistractor
    elif env_name == 'PandaPickPlaceDistractor':
        from robosuite_env.tasks.pick_place import PandaPickPlaceDistractor
        env = PandaPickPlaceDistractor
    elif env_name == 'PandaNutAssemblyDistractor':
        from robosuite_env.tasks.nut_assembly import PandaNutAssemblyDistractor
        env = PandaNutAssemblyDistractor
    elif env_name == 'SawyerNutAssemblyDistractor':
        from robosuite_env.tasks.nut_assembly import SawyerNutAssemblyDistractor
        env = SawyerNutAssemblyDistractor
    elif env_name == 'PandaBlockStacking':
        from robosuite_env.tasks.stack import PandaBlockStacking
        env = PandaBlockStacking
    elif env_name == 'SawyerBlockStacking':
        from robosuite_env.tasks.stack import SawyerBlockStacking
        env = SawyerBlockStacking
    elif env_name == 'PandaBasketball':
        from robosuite_env.tasks.basketball import PandaBasketball
        env = PandaBasketball
    elif env_name == 'SawyerBasketball':
        from robosuite_env.tasks.basketball import SawyerBasketball
        env = SawyerBasketball
    elif env_name == 'PandaInsert':
        from robosuite_env.tasks.insert import PandaInsert
        env = PandaInsert
    elif env_name == 'SawyerInsert':
        from robosuite_env.tasks.insert import SawyerInsert
        env = SawyerInsert
    elif env_name == 'PandaDrawer':
        from robosuite_env.tasks.drawer import PandaDrawer
        env = PandaDrawer
    elif env_name == 'SawyerDrawer':
        from robosuite_env.tasks.drawer import SawyerDrawer
        env = SawyerDrawer
    elif env_name == 'PandaButton':
        from robosuite_env.tasks.press_button import PandaButton
        env = PandaButton
    elif env_name == 'SawyerButton':
        from robosuite_env.tasks.press_button import SawyerButton
        env = SawyerButton
    elif env_name == 'PandaDoor':
        from robosuite_env.tasks.door import PandaDoor
        env = PandaDoor
    elif env_name == 'SawyerDoor':
        from robosuite_env.tasks.door import SawyerDoor
        env = SawyerDoor
    else:
        raise NotImplementedError
    
    from robosuite_env.custom_ik_wrapper import CustomIKWrapper
    return CustomIKWrapper(env(**kwargs), ranges=ranges)
