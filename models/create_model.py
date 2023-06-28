from .vit import ViT
from .swin import SwinTransformer
from .cait import CaiT
from .pit import PiT
from .t2t import T2T_ViT

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        if img_size == 32:
            patch_size = 4
        elif img_size == 64:
            patch_size = 8
        elif img_size == 256:
            patch_size = 32

        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit-base':
        patch_size = 2 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=256, 
                    mlp_dim_ratio=2, depth=8, heads=4, dim_head=256//4,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit-heat':
        patch_size = 2 if img_size == 32 else 4
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192,
                    mlp_dim_ratio=2, depth=8, heads=6, dim_head=192//6,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit_d6':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=252, 
                    mlp_dim_ratio=2, depth=6, heads=12, dim_head=252//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)

    elif args.model == 'vit_d15':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=144, 
                    mlp_dim_ratio=2, depth=15, heads=12, dim_head=144//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit_d30':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=72, 
                    mlp_dim_ratio=2, depth=30, heads=12, dim_head=72//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit_d30*':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=108, 
                    mlp_dim_ratio=2, depth=30, heads=12, dim_head=108//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)
        
    elif args.model == 'vit_d30**':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=30, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)     
        
    elif args.model == 'vit_d60':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=72, 
                    mlp_dim_ratio=2, depth=60, heads=12, dim_head=72//12,
                    stochastic_depth=args.sd, is_GMM=args.is_GMM, num_kernals=args.num_kernals)
        
    elif args.model =='swin':
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_GMM=args.is_GMM, num_kernals=args.num_kernals)
    
    elif args.model == 'cait':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=args.sd, is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals,
                     )

    elif args.model == 'cait-tiny':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(dim=108, img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=args.sd, is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals,
                     )
        
    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4    
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, 
                    stochastic_depth=args.sd,
                    is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals
                    )
        
    elif args.model =='t2t':
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=args.sd, 
                        is_GMM=args.is_GMM, is_SLM=args.is_SLM, num_kernals=args.num_kernals
                        )
    
    return model